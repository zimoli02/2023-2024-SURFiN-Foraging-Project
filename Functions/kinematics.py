import numpy as np
import pandas as pd
import torch

from typing import Union
import os
from dotmap import DotMap

import sys
from pathlib import Path

current_script_path = Path(__file__).resolve()
parent_dir = current_script_path.parent.parent
sys.path.insert(0, str(parent_dir))

import aeon
from aeon.schema.dataset import exp02

import Functions.inference as inference
import Functions.learning as learning


def NestError(last_positions, next_positions):
    last_x, last_y = last_positions[0], last_positions[1]
    next_x, next_y = next_positions[0], next_positions[1]
    
    if last_x <= 250 and next_x <= 250: return True
    else: return False

def WallError(last_positions, next_positions):
    last_x, last_y = last_positions[0], last_positions[1]
    next_x, next_y = next_positions[0], next_positions[1]
    
    if abs(last_x - next_x) <= 5 and abs(last_y - next_y) <= 5 : return True
    else: return False


def FixNan(mouse_pos):
    df = mouse_pos.copy()
    nan_blocks = df['x'].isna()

    for group, data in mouse_pos[nan_blocks].groupby((nan_blocks != nan_blocks.shift()).cumsum()):
        latest_valid_index = mouse_pos.loc[:data.index[0]-pd.Timedelta('0.018S'), 'x'].last_valid_index()
        latest_valid_values = mouse_pos.loc[latest_valid_index, ['x', 'y']].values
        
        if len(data) == 1:
            df.loc[data.index, 'x'] = latest_valid_values[0]
            df.loc[data.index, 'y'] = latest_valid_values[1]
            
        else:    
            next_valid_index = mouse_pos.loc[data.index[-1]+pd.Timedelta('0.018S'):].first_valid_index()
            next_valid_values = mouse_pos.loc[next_valid_index, ['x', 'y']].values
            
            duration = (data.index[-1] - latest_valid_index).total_seconds()
            interpolated_times = (data.index - latest_valid_index).total_seconds() / duration
                        
            total_x = next_valid_values[0] - latest_valid_values[0]
            total_y = next_valid_values[1] - latest_valid_values[1]
                        
            df.loc[data.index, 'x'] = latest_valid_values[0] + interpolated_times * total_x
            df.loc[data.index, 'y'] = latest_valid_values[1] + interpolated_times * total_y
    
    return df

def FixNestError(mouse_pos, nest_upper = 575, nest_lower = 475):
    df = mouse_pos.copy()
    nest_blocks = df['x'] <= 250
    for group, data in df[nest_blocks].groupby((nest_blocks != nest_blocks.shift()).cumsum()):
        df.loc[(df.index.isin(data.index)) & (df['y'] > nest_upper), 'y'] = nest_upper
        df.loc[(df.index.isin(data.index)) & (df['y'] < nest_lower), 'y'] = nest_lower
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


def ProcessRawData(mouse_pos, root, start, end, exclude_maintenance = True, fix_nan = True, fix_nest = True):
    if exclude_maintenance: mouse_pos = ExcludeMaintenanceData(mouse_pos, GetExperimentTimes(root, start, end))
    
    temp_df = mouse_pos.dropna(subset=['x', 'y'])
    first_valid_index, last_valid_index = temp_df.index[0], temp_df.index[-1]
    mouse_pos = mouse_pos.loc[first_valid_index:last_valid_index]
    
    if fix_nan: mouse_pos = FixNan(mouse_pos)
    if fix_nest: mouse_pos = FixNestError(mouse_pos)
    
    return mouse_pos
    

def LDSParameters_Manual(dt):
    pos_x0, pos_y0 = 0, 0
    vel_x0, vel_y0 = 0.0, 0.0
    acc_x0, acc_y0 = 0.0, 0.0

    # Manual Parameters
    sigma_a = 1.3
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


    Qe = np.array([[dt**4/4, dt**3/2, dt**2/2, 0, 0, 0],
                   [dt**3/2, dt**2,   dt,      0, 0, 0],
                   [dt**2/2, dt,      1,       0, 0, 0],
                   [0, 0, 0, dt**4/4, dt**3/2, dt**2/2],
                   [0, 0, 0, dt**3/2, dt**2,   dt],
                   [0, 0, 0, dt**2/2, dt,      1]],
                dtype=np.double)
    Q = sigma_a**2 * Qe

    sigma_x = 1
    sigma_y = 1

    Z = np.array([[1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0]],
                dtype=np.double)
    R = np.diag([sigma_x**2, sigma_y**2])
    
    return sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, V0, Z, R


def LDSParameters_Learned(y, sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, Z, dt = 0.02):
    '''   
    pos_x0, pos_y0 = y[0,0], y[1,0]
    vel_x0, vel_y0 = (y[0,1] - y[0,0])/dt, (y[1,1] - y[1,0])/dt
    acc_x0, acc_y0 = (y[0,2] - y[0,0])/(dt**2), (y[1,2] - y[1,0])/(dt**2)
    '''
    
    # Learning Parameters
    lbfgs_max_iter = 2
    lbfgs_tolerance_grad = 1e-3
    lbfgs_tolerance_change = 1e-3
    lbfgs_lr = 1.0
    lbfgs_n_epochs = 100
    lbfgs_tol = 1e-3
    
    Qe_reg_param_learned = 1e-10
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
    m0 = optim_res_learned["estimates"]["m0"].numpy()
    V0 = np.diag(sqrt_diag_V0_value**2)
    R = np.diag(optim_res_learned["estimates"]["sqrt_diag_R"].numpy()**2)

    return sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value[0], B, m0, V0, Z, R


def AddKinematics(title, mouse_pos):
    smoothRes = np.load('../Data/ProcessedMouseKinematics/' + title+'smoothRes.npz')
    mouse_pos['smoothed_position_x'] = pd.Series(smoothRes['xnN'][0][0], index=mouse_pos.index)
    mouse_pos['smoothed_position_y'] = pd.Series(smoothRes['xnN'][3][0], index=mouse_pos.index)
    mouse_pos['smoothed_velocity_x'] = pd.Series(smoothRes['xnN'][1][0], index=mouse_pos.index)
    mouse_pos['smoothed_velocity_y'] = pd.Series(smoothRes['xnN'][4][0], index=mouse_pos.index)
    mouse_pos['smoothed_acceleration_x'] = pd.Series(smoothRes['xnN'][2][0], index=mouse_pos.index)
    mouse_pos['smoothed_acceleration_y'] = pd.Series(smoothRes['xnN'][5][0], index=mouse_pos.index)

    x_vel, y_vel = mouse_pos['smoothed_velocity_x'], mouse_pos['smoothed_velocity_y']
    vel = np.sqrt(x_vel**2 + y_vel**2)
    mouse_pos['smoothed_speed'] = pd.Series(vel)
        
    x_acc, y_acc = mouse_pos['smoothed_acceleration_x'], mouse_pos['smoothed_acceleration_y']
    acc = np.sqrt(x_acc**2 + y_acc**2)
    mouse_pos['smoothed_acceleration'] = pd.Series(acc)

def AddKinematics_filter(mouse_pos, filterRes):
    mouse_pos['filtered_position_x'] = pd.Series(filterRes['xnn'][0][0], index=mouse_pos.index)
    mouse_pos['filtered_position_y'] = pd.Series(filterRes['xnn'][3][0], index=mouse_pos.index)
    mouse_pos['filtered_velocity_x'] = pd.Series(filterRes['xnn'][1][0], index=mouse_pos.index)
    mouse_pos['filtered_velocity_y'] = pd.Series(filterRes['xnn'][4][0], index=mouse_pos.index)
    mouse_pos['filtered_acceleration_x'] = pd.Series(filterRes['xnn'][2][0], index=mouse_pos.index)
    mouse_pos['filtered_acceleration_y'] = pd.Series(filterRes['xnn'][5][0], index=mouse_pos.index)

    x_vel, y_vel = mouse_pos['filtered_velocity_x'], mouse_pos['filtered_velocity_y']
    vel = np.sqrt(x_vel**2 + y_vel**2)
    mouse_pos['filtered_speed'] = pd.Series(vel)
        
    x_acc, y_acc = mouse_pos['filtered_acceleration_x'], mouse_pos['filtered_acceleration_y']
    acc = np.sqrt(x_acc**2 + y_acc**2)
    mouse_pos['filtered_acceleration'] = pd.Series(acc)