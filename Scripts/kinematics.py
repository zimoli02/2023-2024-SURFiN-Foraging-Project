import numpy as np
import pandas as pd
import inference
import learning
import torch

from typing import Union
import os
from dotmap import DotMap

import aeon
from aeon.schema.dataset import exp02

def FixNan(mouse_pos):
    df = mouse_pos.copy()
    nan_blocks = df['x'].isna()
    for group, data in df[nan_blocks].groupby((nan_blocks != nan_blocks.shift()).cumsum()):
        duration = data.index[-1] - data.index[0]
        if duration.total_seconds() >= 3:
            latest_valid_index = df.loc[:data.index[0], 'x'].last_valid_index()
            if latest_valid_index is not None:
                latest_valid_values = df.loc[latest_valid_index, ['x', 'y']].values
                df.loc[data.index, ['x', 'y']] = np.tile(latest_valid_values, (len(data.index), 1))
    return df

def GetExperimentTimes(
    root: Union[str, os.PathLike], start_time: pd.Timestamp, end_time: pd.Timestamp
) -> DotMap:
    """
    Retrieve experiment start and stop times from environment states
    (i.e. times outside of maintenance mode) occurring within the
    given start and end times.

    Args:
        root (str or os.PathLike): The root path where epoch data is stored.
        start_time (pandas.Timestamp): Start time.
        end_time (pandas.Timestamp): End time.

    Returns:
        DotMap: A DotMap object containing two keys: 'start' and 'stop',
        corresponding to pairs of experiment start and stop times.

    Notes:
    This function uses the last 'Maintenance' event as the last 'Experiment'
    stop time. If the first retrieved state is 'Maintenance' (e.g.
    'Experiment' mode entered before `start`), `start` is used
    as the first 'Experiment' start time.
    """

    experiment_times = DotMap()
    env_states = aeon.load(
        root,
        exp02.ExperimentalMetadata.EnvironmentState,
        start_time,
        end_time,
    )
    # Use the last 'maintenance' event as end time
    end_time = (env_states[env_states.state == "Maintenance"]).index[-1]
    env_states = env_states[~env_states.index.duplicated(keep="first")]
    # Retain only events between visit start and stop times
    env_states = env_states.iloc[
        env_states.index.get_indexer([start_time], method="bfill")[
            0
        ] : env_states.index.get_indexer([end_time], method="ffill")[0] + 1
    ]
    # Retain only events where state changes (experiment-maintenance pairs)
    env_states = env_states[env_states["state"].ne(env_states["state"].shift())]
    if env_states["state"].iloc[0] == "Maintenance":
        # Pad with an "Experiment" event at the start
        env_states = pd.concat(
            [
                pd.DataFrame(
                    "Experiment",
                    index=[start_time],
                    columns=env_states.columns,
                ),
                env_states,
            ]
        )
    else:
        # Use start time as the first "Experiment" event
        env_states.rename(index={env_states.index[0]: start_time}, inplace=True)
    experiment_times.start = env_states[
        env_states["state"] == "Experiment"
    ].index.values
    experiment_times.stop = env_states[
        env_states["state"] == "Maintenance"
    ].index.values

    return experiment_times


def ExcludeMaintenanceData(
    data: pd.DataFrame, experiment_times: DotMap
) -> pd.DataFrame:
    """
    Exclude rows not in experiment times (i.e., corresponding to maintenance times)
    from the given dataframe.

    Args:
        data (pandas.DataFrame): The data to filter. Expected to have a DateTimeIndex.
        experiment_times (DotMap): A DotMap object containing experiment start and stop times.

    Returns:
        pandas.DataFrame: The filtered data.
    """
    filtered_data = pd.concat(
        [
            data.loc[start:stop]
            for start, stop in zip(experiment_times.start, experiment_times.stop)
        ]
    )
    return filtered_data


def ProcessRawData(mouse_pos, root, start, end):
    temp_df = mouse_pos.dropna(subset=['x', 'y'])
    first_valid_index = temp_df.index[0]
    last_valid_index = temp_df.index[-1]
    mouse_pos = mouse_pos.loc[first_valid_index:last_valid_index]
    
    mouse_pos = FixNan(mouse_pos)
    experiment_times = GetExperimentTimes(root, start, end)
    mouse_pos = ExcludeMaintenanceData(mouse_pos, experiment_times)
    
    return mouse_pos
    

def LDSParameters_Manual(dt):
    pos_x0, pos_y0 = 0, 0
    vel_x0, vel_y0 = 0.0, 0.0
    acc_x0, acc_y0 = 0.0, 0.0

    # Manual Parameters
    sigma_a = 0.5
    sqrt_diag_V0_value = 1e-3

    m0 = np.array([pos_x0, vel_x0, acc_x0, pos_y0, vel_y0, acc_y0], dtype=np.double)
    V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)


    B = np.array([[1, dt, dt**2/2, 0, 0, 0],
                [0, 1, dt, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, dt, dt**2/2],
                [0, 0, 0, 0, 1, dt],
                [0, 0, 0, 0, 0, 1]],
                dtype=np.double)


    Qe = np.array([[dt**5/20, dt**4/8, dt**3/6, 0, 0, 0],
                [dt**4/8, dt**3/3,  dt**2/2, 0, 0, 0],
                [dt**3/6, dt**2/2,  dt,      0, 0, 0],
                [0, 0, 0,                    dt**5/20, dt**4/8, dt**3/6],
                [0, 0, 0,                    dt**4/8, dt**3/3,  dt**2/2],
                [0, 0, 0,                    dt**3/6, dt**2/2,  dt]],
                dtype=np.double)
    Q = sigma_a**2 * Qe

    sigma_x = 1
    sigma_y = 1

    Z = np.array([[1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0]],
                dtype=np.double)
    R = np.diag([sigma_x**2, sigma_y**2])
    
    return sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, V0, Z, R


def LDSParameters_Learned(y, dt, sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, Z):
    pos_x0, pos_y0 = y[0,0], y[1,0]
    vel_x0, vel_y0 = 0.0, 0.0
    acc_x0, acc_y0 = 0.0, 0.0
    
    # Learning Parameters
    lbfgs_max_iter = 2
    lbfgs_tolerance_grad = -1
    lbfgs_tolerance_change = 1e-2
    lbfgs_lr = 0.01
    lbfgs_n_epochs = 75
    lbfgs_tol = 1e-2
    
    Qe_reg_param_learned = 1e-2
    sqrt_diag_R_torch = torch.DoubleTensor([sigma_x, sigma_y])
    m0_torch = torch.from_numpy(m0.copy())
    sqrt_diag_V0_torch = torch.DoubleTensor([sqrt_diag_V0_value
                                            for i in range(len(m0))])
    if Qe_reg_param_learned is not None:
        Qe_regularized_learned = Qe + Qe_reg_param_learned * np.eye(Qe.shape[0])
    else:
        Qe_regularized_learned = Qe
    y_torch = torch.from_numpy(y.astype(np.double))
    B_torch = torch.from_numpy(B.astype(np.double))
    Qe_regularized_learned_torch = torch.from_numpy(Qe_regularized_learned.astype(np.double))
    Z_torch = torch.from_numpy(Z.astype(np.double))

    vars_to_estimate = {}
    vars_to_estimate["sigma_a"] = True
    vars_to_estimate["sqrt_diag_R"] = True
    vars_to_estimate["R"] = True
    vars_to_estimate["m0"] = True
    vars_to_estimate["sqrt_diag_V0"] = True
    vars_to_estimate["V0"] = True

    optim_res_learned = learning.torch_lbfgs_optimize_SS_tracking_diagV0(
        y=y_torch, B=B_torch, sigma_a0=sigma_a,
        Qe=Qe_regularized_learned_torch, Z=Z_torch, sqrt_diag_R_0=sqrt_diag_R_torch, m0_0=m0_torch,
        sqrt_diag_V0_0=sqrt_diag_V0_torch, max_iter=lbfgs_max_iter, lr=lbfgs_lr,
        vars_to_estimate=vars_to_estimate, tolerance_grad=lbfgs_tolerance_grad,
        tolerance_change=lbfgs_tolerance_change, n_epochs=lbfgs_n_epochs,
        tol=lbfgs_tol)
    
    sigma_a = optim_res_learned["estimates"]["sigma_a"].item()
    sigma_x = optim_res_learned["estimates"]["sqrt_diag_R"].numpy()[0]
    sigma_y = optim_res_learned["estimates"]["sqrt_diag_R"].numpy()[1]
    sqrt_diag_V0_value = optim_res_learned["estimates"]["sqrt_diag_V0"].numpy()
    Q = sigma_a**2*Qe
    m0 = optim_res_learned["estimates"]["m0"].numpy()
    V0 = np.diag(sqrt_diag_V0_value**2)
    R = np.diag(optim_res_learned["estimates"]["sqrt_diag_R"].numpy()**2)

    return sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value[0], B, Q, m0, V0, Z, R