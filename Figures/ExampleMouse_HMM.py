import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

import sys
from pathlib import Path

aeon_mecha_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(aeon_mecha_dir))

import aeon
import aeon.io.api as api
from aeon.schema.dataset import exp02
from aeon.analysis.utils import visits

import SSM.ssm as ssm
from SSM.ssm.util import find_permutation
from SSM.ssm.plots import gradient_cmap, white_to_color_cmap

import Functions.HMM as HMM
import Functions.kinematics as kinematics
import Functions.patch as patch


root = [Path("/ceph/aeon/aeon/data/raw/AEON2/experiment0.2")]

subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])
short_sessions = sessions.iloc[[4,16,17,20,23,24,25,28,29,30,31]]
long_sessions = sessions.iloc[[8, 10, 11, 14]]
example_sessions = [1,3,4]

feature = ['smoothed_speed', 'smoothed_acceleration']
color_names = ['black', "blue", "red", "tan", "green", "brown", "purple", "orange", "black", 'turquoise']
state_names = ['Resting', 'Eating', 'Digging', 'Clinging', 'Exploring', 'Fast-Leaving']
alpha = [0.4, 0.4, 0.2, 0.4, 0.7, 0.9]

scale = 2e-3

def FindModels():
    LogLikelihood = dict()
    for key in example_sessions:
        LogLikelihood[key] = []
    
    session_number = len(example_sessions)
    for N in range(3,17):
        fig, axs = plt.subplots(session_number, 1, figsize = (N, 4*session_number))
        width = 0.4
        for j in range(len(example_sessions)):
            title = 'ShortSession'+str(example_sessions[j])
            print(title)
            
            mouse_pos = pd.read_parquet('../Data/MousePos/' + title + 'mousepos.parquet', engine='pyarrow')
            obs = np.array(mouse_pos[feature])
        
            hmm, states, transition_mat, lls = HMM.FitHMM(obs, num_states = N, n_iters = 50)
            ll = hmm.log_likelihood(obs)
            LogLikelihood[example_sessions[j]].append(ll)
            
            speed_index = np.argsort(hmm.observations.params[0].T[0], -1)   
            mean_speed, mean_acce = hmm.observations.params[0].T[0][speed_index], hmm.observations.params[0].T[1][speed_index]
            var_speed, var_acce = hmm.observations.params[1][speed_index][:,0,0], hmm.observations.params[1][speed_index][:,1,1]
            

            axs[j].bar(np.arange(N)-width/2, mean_speed, yerr=var_speed, width = width, capsize=5, color = 'blue', label = 'Speed')
            axs[j].bar(np.arange(N)+width/2, mean_acce, yerr=var_acce, width = width, capsize=5, color = 'orange', label = 'Acce')
            axs[j].set_xticks(range(0, N), [str(i) for i in range(N)])
            axs[j].set_ylabel('Feature')
            axs[j].legend()
        plt.savefig('../Figures/Results/StateNumber' + str(N) +'.png')
        plt.show()
    LogLikelihood = np.array([LogLikelihood], dtype=object)
    np.save('../Figures/Results/LogLikelihood.npy', LogLikelihood)

def DisplayLikelihood():
    N = np.arange(3,17,1)
    LogLikelihood = np.load('../Figures/Results/LogLikelihood.npy', allow_pickle=True)
    LogLikelihood = LogLikelihood.item()
    
    fig, axs = plt.subplots(1,1,figsize = (10,7))
    for key in LogLikelihood.keys():
        LL = LogLikelihood[key]
        axs.scatter(N, LL)
        axs.plot(N, LL, label = 'Session ' + str(key))
    axs.legend(loc = 'lower right')
    axs.set_xticks(N)
    
    plt.tight_layout()
    plt.savefig('../Figures/Results/LogLikelihood.png')
    plt.show()

def FitExampleSession(id, n):
    title = 'ShortSession' + str(id)
    mouse_pos = pd.read_parquet('../Data/MousePos/' + title + 'mousepos.parquet', engine='pyarrow')
    
    obs = np.array(mouse_pos[feature])
    hmm, states, transition_mat, lls = HMM.FitHMM(obs, num_states = n-1, n_iters = 50)
    speed_index = np.argsort(hmm.observations.params[0].T[0], -1) 
    mean_speed, mean_acce = hmm.observations.params[0].T[0][speed_index], hmm.observations.params[0].T[1][speed_index]
    var_speed, var_acce = hmm.observations.params[1][speed_index][:,0,0], hmm.observations.params[1][speed_index][:,1,1]
    
    width = 0.4
    fig, axs = plt.subplots(2, 1, figsize = (n-1, 4*2))
    axs[0].bar(np.arange(n-1), mean_speed, yerr=var_speed**0.5, width = width, capsize=5, color = 'blue')
    axs[1].bar(np.arange(n-1), mean_acce, yerr=var_acce**0.5, width = width, capsize=5, color = 'orange')
    axs[0].set_xticks(range(n-1), [str(i) for i in range(1,n)])
    axs[1].set_xticks(range(n-1), [str(i) for i in range(1,n)])
    axs[0].set_ylabel('Speed')
    axs[1].set_ylabel('Acceleration')
    
    plt.tight_layout()
    plt.savefig('../Figures/Results/ExampleSessionParameters.png')
    plt.show()
    
    np.save('../Figures/Results/ExampleTransM.npy', transition_mat[speed_index].T[speed_index].T)
    
    new_values = np.empty_like(states)
    for i, val in enumerate(speed_index): new_values[states == val] = i
    states = new_values
    states += 1
    
    mouse_pos = patch.PositionInPatch(mouse_pos, r = 100)
    for i in range(len(states)):
        if states[i] == 1:
            if mouse_pos.Patch[i] == 0: states[i] = 0
    
    np.save('../Figures/Results/States.npy', states)
    
def GetStates(id, n, denoise = True):
    title = 'ShortSession' + str(id)
    mouse_pos = pd.read_parquet('../Data/MousePos/' + title + 'mousepos.parquet', engine='pyarrow')
    states = np.load('../Figures/Results/States.npy', allow_pickle=True)
    mouse_pos['states'] = pd.Series(states, index = mouse_pos.index)
    
    mouse_pos.smoothed_velocity_x = mouse_pos.smoothed_velocity_x * scale
    mouse_pos.smoothed_acceleration_x = mouse_pos.smoothed_acceleration_x * scale
    mouse_pos.smoothed_speed = mouse_pos.smoothed_speed * scale
    mouse_pos.smoothed_acceleration = mouse_pos.smoothed_acceleration * scale
    
    if denoise:
        resampled_mouse_pos = mouse_pos.resample('1S').agg({
        'x': 'mean',  
        'y': 'mean',  
        'smoothed_velocity_x':'mean',
        'smoothed_acceleration_x':'mean',
        'smoothed_position_x':'mean',
        'smoothed_position_y':'mean',
        'smoothed_speed':'mean',
        'smoothed_acceleration':'mean',
        'states': lambda x: x.mode()[0] if not x.empty else None  # Most frequent value in 'states'
        })
    
        return resampled_mouse_pos
    else: 
        return mouse_pos

def ManualTransM(id, n, denoise):
    mouse_pos = GetStates(id, n, denoise=denoise)
    states = mouse_pos['states'].to_numpy()

    TransM = np.zeros((n,n))
    
    for i in range(len(states)-1):
        TransM[states[i]][states[i+1]] += 1
        
    for i in range(n):
        TransM[i] = TransM[i]/sum(TransM[i])
    
    return TransM

def DisplayExampleSessionTransM(id, n, denoise = True):
    if not denoise:
        TransM = np.load('../Figures/Results/ExampleTransM.npy', allow_pickle=True)
    else:
        TransM = ManualTransM(id, n, denoise = denoise)
    HMM.PlotTransition(TransM, title = '../Figures/Results/ExampleTransM.png')
    

def DisplayExampleSessionPosition(id, n, denoise = True):
    mouse_pos = GetStates(id, n, denoise=denoise)
    
    x = mouse_pos['smoothed_position_x'].to_numpy()
    y = mouse_pos['smoothed_position_y'].to_numpy()
    states = mouse_pos['states'].to_numpy()
    
    fig, axs = plt.subplots(1, n, figsize = (n*8-2,6))
    for i in range(n):
        axs[i].scatter(x[states == i], y[states == i], color = color_names[i], s = 2, alpha = alpha[i])
        axs[i].set_xlim(145, 1250)
        axs[i].set_ylim(50, 1080)
        #axs[i].set_title('State ' + str(i))
        axs[i].set_aspect('equal', adjustable='box')
        axs[i].set_title(state_names[i], fontsize = 20)
        axs[i].set_xlabel('X (px)')
    axs[0].set_ylabel('Y (px)')
    
    plt.tight_layout()
    plt.savefig('../Figures/Results/ExamplePositionMap.png')
    plt.show()
    
def DisplayExampleSessionPosition_Speed(id, n, denoise = True):
    mouse_pos = GetStates(id, n, denoise=denoise)
    
    x = mouse_pos['smoothed_position_x'].to_numpy()
    y = mouse_pos['smoothed_position_y'].to_numpy()
    states = mouse_pos['states'].to_numpy()
    
    X_vel = mouse_pos['smoothed_velocity_x'].to_numpy()
    flag = X_vel > 0
    
    fig, axs = plt.subplots(1, n, figsize = (n*8-2,6))
    for i in range(n):
        x_vel_dir = flag[states == i]
        colors = ['blue' if j else 'red' for j in x_vel_dir]
        axs[i].scatter(x[states == i], y[states == i], color = colors, s = 2, alpha = alpha[i])
        axs[i].set_xlim(145, 1250)
        axs[i].set_ylim(50, 1080)
        axs[i].set_aspect('equal', adjustable='box')
        axs[i].set_title(state_names[i], fontsize = 20)
        axs[i].set_xlabel('X (px)')
    axs[0].set_ylabel('Y (px)')
    
    plt.tight_layout()
    plt.savefig('../Figures/Results/ExamplePositionMap_Speed.png')
    plt.show()

def DisplayExampleSessionPosition_Acce(id, n, denoise = True):
    mouse_pos = GetStates(id, n, denoise=denoise)
    
    x = mouse_pos['smoothed_position_x'].to_numpy()
    y = mouse_pos['smoothed_position_y'].to_numpy()
    states = mouse_pos['states'].to_numpy()
    
    X_acce = mouse_pos['smoothed_acceleration_x'].to_numpy()
    flag = X_acce > 0
    
    fig, axs = plt.subplots(1, n, figsize = (n*8-2,6))
    for i in range(n):
        x_acce_dir = flag[states == i]
        colors = ['green' if j else 'purple' for j in x_acce_dir]
        axs[i].scatter(x[states == i], y[states == i], color = colors, s = 2, alpha = alpha[i])
        axs[i].set_xlim(145, 1250)
        axs[i].set_ylim(50, 1080)
        axs[i].set_aspect('equal', adjustable='box')
        axs[i].set_title(state_names[i], fontsize = 20)
        axs[i].set_xlabel('X (px)')
    axs[0].set_ylabel('Y (px)')
    
    plt.tight_layout()
    plt.savefig('../Figures/Results/ExamplePositionMap_Acce.png')
    plt.show()

def Pellets(id, n):
    title = 'ShortSession' + str(id)
    mouse_pos = pd.read_parquet('../Data/MousePos/' + title + 'mousepos.parquet', engine='pyarrow')
    
    start, end = mouse_pos.index[0], mouse_pos.index[-1]
                
    pellets_patch1 = api.load(root, exp02.Patch1.DeliverPellet, start=start, end=end)
    pellets_patch2 = api.load(root, exp02.Patch2.DeliverPellet, start=start, end=end)
    
    states = np.load('../Figures/Results/States.npy', allow_pickle=True)
    mouse_pos['states'] = pd.Series(states, index = mouse_pos.index)
    
    Pellets_State = []

    for i in range(len(pellets_patch1)):
        trigger = pellets_patch1.index[i]
        latest_valid_index = mouse_pos.loc[trigger - pd.Timedelta('6S'):trigger, 'states'].index
        latest_valid_state = mouse_pos.loc[latest_valid_index, ['states']].values.reshape(-1)
        if len(latest_valid_state) >= 50: latest_valid_state  = latest_valid_state[-50:]
        next_valid_index = mouse_pos.loc[trigger:trigger + pd.Timedelta('6S'), 'states'].index
        next_valid_state = mouse_pos.loc[next_valid_index, ['states']].values.reshape(-1)
        if len(next_valid_state) >= 50: next_valid_state  = next_valid_state[:50]
        state = np.concatenate((latest_valid_state, next_valid_state))
        
        Pellets_State.append(state)

    for i in range(len(pellets_patch2)):
        trigger = pellets_patch2.index[i]
        latest_valid_index = mouse_pos.loc[trigger - pd.Timedelta('6S'):trigger, 'states'].index
        latest_valid_state = mouse_pos.loc[latest_valid_index, ['states']].values.reshape(-1)
        if len(latest_valid_state) >= 50: latest_valid_state  = latest_valid_state[-50:]
        next_valid_index = mouse_pos.loc[trigger:trigger + pd.Timedelta('6S'), 'states'].index
        next_valid_state = mouse_pos.loc[next_valid_index, ['states']].values.reshape(-1)
        if len(next_valid_state) >= 50: next_valid_state  = next_valid_state[:50]
        state = np.concatenate((latest_valid_state, next_valid_state))
    
        Pellets_State.append(state)
    
    N = n
    colors = sns.xkcd_palette(color_names[0:N])
    cmap = gradient_cmap(colors)

    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    sns.heatmap(Pellets_State,cmap=cmap, ax=axs, vmin=0, vmax = N-1, cbar = True)
    axs.set_aspect('auto')

    axs.set_xticks([50])
    axs.set_xticklabels(['Pellet'], rotation = 0)

    axs.set_ylabel("Pellet Deliveries")
    axs.set_yticks([])
    
    plt.savefig('../Figures/Results/ExamplePelletStates.png')
    plt.show()
    
def DisplayExampleSessionState(id, n, denoise = True):
    mouse_pos = GetStates(id, n, denoise=denoise)
    start, end = pd.Timestamp('2022-06-21 11:11:20.0'), pd.Timestamp('2022-06-21 11:12:00.0')
    mouse_pos = mouse_pos[start:end]
    HMM.PlotStates(mouse_pos['states'].to_numpy(), mouse_pos, n, title = '../Figures/Results/ExampleStates.png')
    
    
def DataInArena(id, n, denoise):
    mouse_pos = GetStates(id, n, denoise=denoise)
    mouse_pos = patch.PositionInArena(mouse_pos)

    mouse_pos = mouse_pos[mouse_pos['Arena'] == 1]
    
    explore = mouse_pos[mouse_pos['states'] == 4]
    speed_explore, acce_explore = explore.smoothed_speed.to_numpy(), explore.smoothed_acceleration.to_numpy()
    
    leave = mouse_pos[mouse_pos['states'] == 5]
    speed_leave, acce_leave = leave.smoothed_speed.to_numpy(), leave.smoothed_acceleration.to_numpy()
    
    mean_speed = np.array([np.mean(speed_explore), np.mean(speed_leave)])
    var_speed = np.array([np.var(speed_explore), np.var(speed_leave)])
    
    mean_acce = np.array([np.mean(acce_explore), np.mean(acce_leave)])
    var_acce = np.array([np.var(acce_explore), np.var(acce_leave)])
    
    width = 0.15
    x_loc = [0,0.2]
    fig, axs = plt.subplots(2, 1, figsize = (2, 4*2))
    axs[0].bar(x_loc, mean_speed, yerr=var_speed**0.5, width = width, capsize=5, color = 'blue')
    axs[1].bar(x_loc, mean_acce, yerr=var_acce**0.5, width = width, capsize=5, color = 'orange')
    axs[0].set_xticks(x_loc, ['E.', 'F.L.'])
    axs[1].set_xticks(x_loc, ['E.', 'F.L.'])
    axs[0].set_ylabel('Speed (m/s)')
    axs[1].set_ylabel('Acceleration (m/s$^2$)')
    
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('../Figures/Results/ExampleSessionParameters_Arena.png')
    plt.show()
    
def DirectionInArena(id, n, denoise):
    mouse_pos = GetStates(id, n, denoise=denoise)
    mouse_pos = patch.PositionInArena(mouse_pos)
    
    mouse_pos = mouse_pos[mouse_pos['Arena'] == 1]
    
    explore = mouse_pos[mouse_pos['states'] == 4]
    speed_explore_x, acce_explore_x = explore.smoothed_velocity_x.to_numpy(), explore.smoothed_acceleration_x.to_numpy()
    
    leave = mouse_pos[mouse_pos['states'] == 5]
    speed_leave_x, acce_leave_x = leave.smoothed_velocity_x.to_numpy(), leave.smoothed_acceleration_x.to_numpy()
    
    speed_explore_dir = np.where(speed_explore_x < 0, -1, 0) + np.where(speed_explore_x > 0, 1, 0)
    acce_explore_dir = np.where(acce_explore_x < 0, -1, 0) + np.where(acce_explore_x > 0, 1, 0)
    speed_leave_dir = np.where(speed_leave_x < 0, -1, 0) + np.where(speed_leave_x > 0, 1, 0)
    acce_leave_dir = np.where(acce_leave_x < 0, -1, 0) + np.where(acce_leave_x > 0, 1, 0)
    
    mean_speed = np.array([np.mean(speed_explore_dir), np.mean(speed_leave_dir)])
    var_speed = np.array([np.var(speed_explore_dir), np.var(speed_leave_dir)])
    
    mean_acce = np.array([np.mean(acce_explore_dir), np.mean(acce_leave_dir)])
    var_acce = np.array([np.var(acce_explore_dir), np.var(acce_leave_dir)])
    
    width = 0.15
    x_loc = [0,0.2]
    fig, axs = plt.subplots(2, 1, figsize = (2, 4*2))
    axs[0].bar(x_loc, mean_speed, yerr=var_speed**0.5, width = width, capsize=5, color = 'blue')
    axs[1].bar(x_loc, mean_acce, yerr=var_acce**0.5, width = width, capsize=5, color = 'orange')
    
    for i in range(2):
        axs[i].set_xticks(x_loc, ['E.', 'F.L.'])
        axs[i].set_yticks(np.array([-1,0, 1]), ['1.0','0.0', '1.0'])
        axs[i].set_ylabel('Direction')
    
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)

    
    plt.tight_layout()
    plt.savefig('../Figures/Results/ExampleSessionDirection_Arena.png')
    plt.show()

def main():
    ID = 1
    N = 6
    #FindModels()
    #DisplayLikelihood()
    #FitExampleSession(id = ID, n = N)
    
    #DisplayExampleSessionPosition(id = ID, n = N, denoise = False)
    #DisplayExampleSessionPosition_Speed(id = ID, n = N, denoise=False)
    #DisplayExampleSessionPosition_Acce(id = ID, n = N, denoise = False)
    #Pellets(id = ID, n=N)
    #DisplayExampleSessionTransM(id = ID, n= N, denoise=True)
    #DisplayExampleSessionState(id = ID, n=N, denoise = True)
    
    #DataInArena(id = ID, n = N, denoise=False)
    DirectionInArena(id = ID, n = N, denoise=False)

    #FitModelsLong()
    
if __name__ == "__main__":
    main()