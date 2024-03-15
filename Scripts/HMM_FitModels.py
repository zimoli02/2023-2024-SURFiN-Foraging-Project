import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

import sys
from pathlib import Path

aeon_mecha_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(aeon_mecha_dir))

import aeon
import aeon.io.api as api
from aeon.schema.dataset import exp02
from aeon.analysis.utils import visits

import Functions.HMM as HMM
import Functions.kinematics as kinematics
import Functions.patch as patch

def main():
    root = [Path("/ceph/aeon/aeon/data/raw/AEON2/experiment0.2")]

    subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
    sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])
    short_sessions = sessions.iloc[[4,16,17,20,23,24,25,26,28,29,30,31]]
    long_sessions = sessions.iloc[[8, 10, 11, 14]]
    
    N = 5
    feature = ['weight', 'smoothed_speed', 'smoothed_acceleration']
    
    X, Y, SPEED, ACCE, VISIT_1, VISIT_2 = [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)]
    HEATMAP = [[] for _ in range(N)]
    
    for session, j in zip(list(short_sessions.itertuples()), range(len(short_sessions))):
        title = 'ShortSession'+str(j)
        
        mouse_pos = pd.read_parquet('../Data/MousePos' + title + 'mousepos.parquet', engine='pyarrow')
    
        obs = np.array(mouse_pos[feature])
        hmm, states, transition_mat, lls = HMM.FitHMM(obs, num_states = N, n_iters = 50)

        state_mean_speed = hmm.observations.params[0].T[1]
        index = np.argsort(state_mean_speed, -1)
                
        HMM.PlotTransition(transition_mat[index].T[index].T, title = '../Images/HMM_TransitionM/' + title+'.png')

        new_values = np.empty_like(states)
        for i, val in enumerate(index): new_values[states == val] = i
        states = new_values
        np.save('../Data/HMMStates/' + title + "States_Single.npy")
            
        x = mouse_pos['smoothed_position_x']
        y = mouse_pos['smoothed_position_y']
        speed = mouse_pos['smoothed_speed']
        acceleration = mouse_pos['smoothed_acceleration']
        VisitPatch1 = mouse_pos['Patch1']
        VisitPatch2 = mouse_pos['Patch2']
            
        for i in range(N):
            X[i] = np.concatenate([X[i], x[states==i]])
            Y[i] = np.concatenate([Y[i], y[states==i]])
            SPEED[i] = np.concatenate([SPEED[i], speed[states==i]])
            ACCE[i] = np.concatenate([ACCE[i], acceleration[states == i]])
            VISIT_1[i] = np.concatenate([VISIT_1[i], VisitPatch1[states == i]])
            VISIT_2[i] = np.concatenate([VISIT_2[i], VisitPatch2[states == i]])
            
        fig, axs = plt.subplots(1, N, figsize = (N*8-2,6))
        for i in range(N):
            heatmap, xedges, yedges, img = axs[i].hist2d(x[states == i], y[states == i], bins=[50, 50], range=[[215, 1235], [65, 1065]], cmap='binary', density=True)
            heatmap = np.nan_to_num(heatmap)
            HEATMAP[i].append(heatmap)
            axs[i].set_xlim(205, 1255)
            axs[i].set_ylim(45, 1085)
            axs[i].set_title('State' + str(i))
            axs[i].set_xlabel('X')
            axs[i].set_ylabel('Y')
        plt.savefig('../Images/HMM_Heatmap/' + title+'.png')


    fig, axs = plt.subplots(2, N, figsize = (N*8-2,14))
    for i in range(N):
        combined_heatmaps = np.array(HEATMAP[i])
        average_heatmap, std_deviation_heatmap = np.mean(combined_heatmaps, axis=0), np.std(combined_heatmaps, axis=0)
        sns.heatmap(np.rot90(average_heatmap), ax=axs[0,i], cmap='binary', cbar = False)
        sns.heatmap(np.rot90(std_deviation_heatmap), ax=axs[1,i], cmap='binary', cbar = False)
        
        axs[0,i].set_xticks([])
        axs[0,i].set_yticks([])
        axs[1,i].set_xticks([])
        axs[1,i].set_yticks([])
        axs[0,i].set_ylabel('Y')
        axs[1,i].set_ylabel('Y')
        axs[1,i].set_xlabel('X')
        axs[0,i].set_title('State' + str(i))    
    plt.savefig('../Images/HMM_Heatmap/Average.png')

    # Speed, Acceleration, Visits in Patch 1, Visits in Patch 2
    fig, axs = plt.subplots(4, 1, figsize = (10, 4*7-1))
    DATA = [SPEED, ACCE, VISIT_1, VISIT_2]
    FEATURE = ['SPEED', 'ACCE', 'VISIT_1', 'VISIT_2']
    for data, i in zip(DATA, range(len(DATA))):
        means = [np.mean(arr) for arr in data]
        std_devs = [np.std(arr) for arr in data]
        axs[i].bar(range(N), means, yerr=std_devs, capsize=5)
        axs[i].set_xticks(range(0, 5), ['0', '1', '2', '3','4'])
        axs[i].set_ylabel(FEATURE[i])
    plt.savefig('../Images/HMM_Data/DataAverage.png')
    
    
if __name__ == "__main__":
    main()