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

feature = ['smoothed_speed', 'smoothed_acceleration', 'r']
color_names = ['black', "blue", "red", "tan", "green", "brown", "purple", "orange", "black", 'turquoise']

def FitModelsShort(n=5):
    N = n

    X, Y, SPEED, ACCE, VISIT_1, VISIT_2 = [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)]
    HEATMAP = [[] for _ in range(N)]
    
    for session, j in zip(list(short_sessions.itertuples()), range(len(short_sessions))):
        title = 'ShortSession'+str(j)
        
        mouse_pos = pd.read_parquet('../Data/MousePos/' + title + 'mousepos.parquet', engine='pyarrow')
    
        obs = np.array(mouse_pos[feature])
        hmm, states, transition_mat, lls = HMM.FitHMM(obs, num_states = N, n_iters = 50)

        state_mean_speed = hmm.observations.params[0].T[1]
        index = np.argsort(state_mean_speed, -1)
                
        HMM.PlotTransition(transition_mat[index].T[index].T, title = '../Images/HMM_TransitionM/' + title+'.png')
        np.save('../Data/HMMStates/' + title + "TransM_Single.npy", transition_mat[index].T[index].T)
        
        new_values = np.empty_like(states)
        for i, val in enumerate(index): new_values[states == val] = i
        states = new_values
        np.save('../Data/HMMStates/' + title + "States_Single.npy", states)
            
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
            
        '''fig, axs = plt.subplots(1, N, figsize = (N*8-2,6))
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
        plt.show()'''
        
        fig, axs = plt.subplots(1, N, figsize = (N*8-2,6))
        for i in range(N):
            axs[i].scatter(x[states == i], y[states == i], color = color_names[i], s = 2, alpha = 0.5)
            axs[i].set_xlim(145, 1250)
            axs[i].set_ylim(50, 1080)
            axs[i].set_aspect('equal', adjustable='box')
            axs[i].set_xlabel('X (px)')
        axs[0].set_ylabel('Y (px)')
        plt.tight_layout()
        plt.savefig('../Images/HMM_Heatmap/' + title+'.png')
        plt.show()



    '''    
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
    plt.savefig('../Images/HMM_Heatmap/ShortAverage.png')
    plt.show()'''

    # Speed, Acceleration, Visits in Patch 1, Visits in Patch 2
    fig, axs = plt.subplots(4, 1, figsize = (10, 4*7-1))
    DATA = [SPEED, ACCE, VISIT_1, VISIT_2]
    FEATURE = ['SPEED', 'ACCE', 'VISIT_1', 'VISIT_2']
    for data, i in zip(DATA, range(len(DATA))):
        means = [np.mean(arr) for arr in data]
        var = [np.std(arr)/np.sqrt(len(arr)) for arr in data]
        axs[i].bar(range(N), means, yerr=var, capsize=5)
        axs[i].set_xticks(range(0, N),[str(i) for i in range(N)])
        axs[i].set_ylabel(FEATURE[i])
    plt.savefig('../Images/HMM_Data/ShortDataAverage.png')
    plt.show()
    
def FitModelsLong(n=8):
    N = n
    for session, j in zip(list(long_sessions.itertuples()), range(len(long_sessions))):
        title = 'LongSession'+str(j)
                                
        start, end = session.enter, session.exit
        mouse = api.load(root, exp02.CameraTop.Position, start=start, end=end)
            
        mouse = kinematics.ProcessRawData(mouse, root, start, end)
            
        patch.AddKinematics(title, mouse)
            
        mouse_pos_subs = patch.SeparateDF(mouse)

        dfs = []
        for mouse_pos_sub in mouse_pos_subs:
            '''
            mouse_pos_sub = mouse_pos_sub[mouse_pos_sub['smoothed_speed'] <= 2000]
            mouse_pos_sub = mouse_pos_sub[mouse_pos_sub['smoothed_acceleration'] <= 60000]
            '''
            mouse_pos_sub = HMM.DeleteRows(mouse_pos_sub)
                
            start, end = mouse_pos_sub.index[0], mouse_pos_sub.index[-1]
                
            weight = api.load(root, exp02.Nest.WeightSubject, start=start, end=end)
            patch.AddWeight(mouse_pos_sub, weight)

            patch.InPatch(mouse_pos_sub)
                
            hours_series = pd.Series(mouse_pos_sub.index.hour, index=mouse_pos_sub.index)
            mouse_pos_sub['CR'] = hours_series.apply(patch.calculate_cr)
            mouse_pos_sub = mouse_pos_sub.loc[:,['weight', 'smoothed_position_x', 'smoothed_position_y', 'smoothed_speed', 'smoothed_acceleration','Patch1','Patch2', 'CR']]
                
            dfs.append(mouse_pos_sub)

        mouse_pos = dfs[0]
        for df in dfs[1:]: mouse_pos = mouse_pos.add(df, fill_value=0)

            
        SPEED, ACCE, VISIT_1, VISIT_2, CR = [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)]
            
        obs = np.array(mouse_pos[feature])
        hmm, states, transition_mat, lls = HMM.FitHMM(obs, num_states = N, n_iters = 50)

        state_mean_speed = hmm.observations.params[0].T[1]
        index = np.argsort(state_mean_speed, -1)
                    
        HMM.PlotTransition(transition_mat[index].T[index].T, title = '../Images/HMM_TransitionM/' + title+'.png')
        np.save('../Data/HMMStates/' + title + "TransM_Single.npy", transition_mat[index].T[index].T)
                    
        new_values = np.empty_like(states)
        for i, val in enumerate(index): new_values[states == val] = i
        states = new_values
        np.save('../Data/HMMStates/' + title + "States_Single.npy", states)
                    
        x = mouse_pos['smoothed_position_x']
        y = mouse_pos['smoothed_position_y']
        speed = mouse_pos['smoothed_speed']
        acceleration = mouse_pos['smoothed_acceleration']
        VisitPatch1 = mouse_pos['Patch1']
        VisitPatch2 = mouse_pos['Patch2']
        cr = mouse_pos['CR']

        for i in range(N):
            SPEED[i] = np.concatenate([SPEED[i], speed[states==i]])
            ACCE[i] = np.concatenate([ACCE[i], acceleration[states == i]])
            VISIT_1[i] = np.concatenate([VISIT_1[i], VisitPatch1[states == i]])
            VISIT_2[i] = np.concatenate([VISIT_2[i], VisitPatch2[states == i]])
            CR[i] = np.concatenate([CR[i], cr[states == i]])
                    
        fig, axs = plt.subplots(1, N, figsize = (N*8-2,6))
        for i in range(N):
            heatmap, xedges, yedges, img = axs[i].hist2d(x[states == i], y[states == i], bins=[50, 50], range=[[215, 1235], [65, 1065]], cmap='binary', density=True)
            axs[i].set_xlim(205, 1255)
            axs[i].set_ylim(45, 1085)
            axs[i].set_title('State' + str(i))
            axs[i].set_xlabel('X')
            axs[i].set_ylabel('Y')
        plt.savefig('../Images/HMM_Heatmap/' + title+'.png')
        plt.show()

    fig, axs = plt.subplots(5, 1, figsize = (10, 5*7-1))
    DATA = [SPEED, ACCE, VISIT_1, VISIT_2, CR]
    FEATURE = ['SPEED', 'ACCE', 'VISIT_1', 'VISIT_2', 'CR']
    for data, i in zip(DATA, range(len(DATA))):
        means = [np.mean(arr) for arr in data]
        std_devs = [np.std(arr) for arr in data]
        axs[i].bar(range(N), means, yerr=std_devs, capsize=5)
        axs[i].set_xticks(range(0, N), [str(i) for i in range(N)]) # Change with number of states!!!
        axs[i].set_ylabel(FEATURE[i])
    plt.savefig('../Images/HMM_Data/LongDataAverage.png')
    plt.show()

def main():
    # For short sessions:
    FitModelsShort(n=7)
    
    # For long sessions
    #FitModelsLong()
    
if __name__ == "__main__":
    main()