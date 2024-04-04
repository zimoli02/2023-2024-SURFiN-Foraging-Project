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
color_names = ["blue","red","tan", "green","brown","purple","darkorange", "black", 'turquoise']

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
    plt.savefig('../Figures/Results/LogLikelihood.png')
    plt.show()

def FitExampleSession(id, n):
    title = 'ShortSession' + str(id)
    mouse_pos = pd.read_parquet('../Data/MousePos/' + title + 'mousepos.parquet', engine='pyarrow')
    
    obs = np.array(mouse_pos[feature])
    hmm, states, transition_mat, lls = HMM.FitHMM(obs, num_states = n, n_iters = 50)
    speed_index = np.argsort(hmm.observations.params[0].T[0], -1) 
    mean_speed, mean_acce = hmm.observations.params[0].T[0][speed_index], hmm.observations.params[0].T[1][speed_index]
    var_speed, var_acce = hmm.observations.params[1][speed_index][:,0,0], hmm.observations.params[1][speed_index][:,1,1]
    
    width = 0.4
    fig, axs = plt.subplots(2, 1, figsize = (n, 4*2))
    axs[0].bar(np.arange(n)-width/2, mean_speed, yerr=var_speed**0.5, width = width, capsize=5, color = 'blue')
    axs[1].bar(np.arange(n)+width/2, mean_acce, yerr=var_acce**0.5, width = width, capsize=5, color = 'orange')
    axs[0].set_xticks(range(0, n), [str(i) for i in range(n)])
    axs[1].set_xticks(range(0, n), [str(i) for i in range(n)])
    axs[0].set_ylabel('Speed')
    axs[1].set_ylabel('Acceleration')
    plt.savefig('../Figures/Results/ExampleSessionParameters.png')
    plt.show()
    
    HMM.PlotTransition(transition_mat[speed_index].T[speed_index].T, title = '../Figures/Results/ExampleTransM.png')
    np.save('../Figures/Results/ExampleTransM.npy', transition_mat[speed_index].T[speed_index].T)
        
    new_values = np.empty_like(states)
    for i, val in enumerate(speed_index): new_values[states == val] = i
    states = new_values
    np.save('../Figures/Results/States.npy', states)
    
    

def DisplayExampleSession(id, n):
    title = 'ShortSession' + str(id)
    mouse_pos = pd.read_parquet('../Data/MousePos/' + title + 'mousepos.parquet', engine='pyarrow')
    
    '''start, end = mouse_pos.index[0], mouse_pos.index[-1]
                
    pellets_patch1 = api.load(root, exp02.Patch1.DeliverPellet, start=start, end=end)
    pellets_patch2 = api.load(root, exp02.Patch2.DeliverPellet, start=start, end=end)'''
    
    states = np.load('../Figures/Results/States.npy', allow_pickle=True)
    x = mouse_pos['smoothed_position_x']
    y = mouse_pos['smoothed_position_y']
    fig, axs = plt.subplots(1, n, figsize = (n*8-2,6))
    for i in range(n):
        axs[i].scatter(x[states == i], y[states == i], color = color_names[i], s = 2, alpha = 0.033 * len(x)/len(x[states == i]))
        axs[i].set_xlim(145, 1250)
        axs[i].set_ylim(50, 1080)
        axs[i].set_title('State' + str(i))
        axs[i].set_xlabel('X')
    axs[0].set_ylabel('Y')
    plt.savefig('../Figures/Results/ExamplePositionMap.png')
    plt.show()

def Pellets(id, n):
    title = 'ShortSession' + str(id)
    mouse_pos = pd.read_parquet('../Data/MousePos/' + title + 'mousepos.parquet', engine='pyarrow')
    
    start, end = mouse_pos.index[0], mouse_pos.index[-1]
                
    pellets_patch1 = api.load(root, exp02.Patch1.DeliverPellet, start=start, end=end)
    pellets_patch2 = api.load(root, exp02.Patch2.DeliverPellet, start=start, end=end)
    
    states = np.load('../Figures/Results/States.npy', allow_pickle=True)
    mouse_pos['states'] = pd.Series(states, index = mouse_pos.index)
    
    Pellets1_State, Pellets2_State = dict(), dict()
    for i in range(n):
        Pellets1_State[i] = 0
        Pellets2_State[i] = 0
        
    for i in range(len(pellets_patch1)):
        trigger = pellets_patch1.index[i]
        latest_valid_index = mouse_pos.loc[:trigger, 'states'].last_valid_index()
        latest_valid_state = mouse_pos.loc[latest_valid_index, ['states']].values[0]
        Pellets1_State[latest_valid_state] += 1

    for i in range(len(pellets_patch2)):
        trigger = pellets_patch2.index[i]
        latest_valid_index = mouse_pos.loc[:trigger, 'states'].last_valid_index()
        latest_valid_state = mouse_pos.loc[latest_valid_index, ['states']].values[0]
        Pellets2_State[latest_valid_state] += 1
    
    print(Pellets1_State)
    print(Pellets2_State)
    


def main():

    #FindModels()
    #DisplayLikelihood()
    #FitExampleSession(id = 1, n = 5)
    #DisplayExampleSession(id = 1, n = 5)
    Pellets(id = 1, n=5)

    #FitModelsLong()
    
if __name__ == "__main__":
    main()