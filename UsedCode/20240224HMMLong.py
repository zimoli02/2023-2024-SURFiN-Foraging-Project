import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, SecondLocator, DateFormatter

import plotly.graph_objects as go
import plotly.express as px

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import scipy
import pyarrow

import inference

import aeon
import aeon.io.api as api
from aeon.io import reader, video
from aeon.schema.dataset import exp02
from aeon.analysis.utils import visits, distancetravelled
from sklearn.preprocessing import StandardScaler

import kinematics
import patch
import HMM

root = [Path("/ceph/aeon/aeon/data/raw/AEON2/experiment0.2")]

def DeleteRows(mouse_pos):
    mouse_pos_reset = mouse_pos.reset_index()


    grouping_var = mouse_pos_reset.groupby(mouse_pos_reset.index // 5).ngroup()
    agg_dict = {col: 'mean' for col in mouse_pos_reset.columns if col != 'time'}
    agg_dict['time'] = 'last'

    mouse_pos_grouped = mouse_pos_reset.groupby(grouping_var).agg(agg_dict)

    mouse_pos_grouped.set_index('time', inplace=True)
    mouse_pos_grouped.index.name = None
    
    return mouse_pos_grouped

def main():
    Exp02Summary = api.load(root, exp02.Metadata).metadata[0].toDict()

    subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
    sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])
    long_sessions = sessions.iloc[[8, 10, 11, 14]]
    
    for session, j in zip(list(long_sessions.itertuples()), range(len(long_sessions))):
        if j == 2:
            title = 'LongSession'+str(j)
            print(title)
                                
            start, end = session.enter, session.exit
            mouse = api.load(root, exp02.CameraTop.Position, start=start, end=end)
            
            mouse = kinematics.ProcessRawData(mouse, root, start, end)
            
            patch.AddKinematics(title, mouse)
            
            mouse_pos_subs = patch.SeparateDF(mouse)

            dfs = []
            for mouse_pos_sub in mouse_pos_subs:
                mouse_pos_sub = mouse_pos_sub[mouse_pos_sub['smoothed_speed'] <= 2000]
                mouse_pos_sub = mouse_pos_sub[mouse_pos_sub['smoothed_acceleration'] <= 60000]
                mouse_pos_sub = DeleteRows(mouse_pos_sub)
                
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

            N = 7
            feature = ['weight', 'smoothed_speed', 'smoothed_acceleration']
            SPEED, ACCE, VISIT_1, VISIT_2, CR = [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)]
            
            obs = np.array(mouse_pos[feature])
            hmm, states, transition_mat, lls = HMM.FitHMM(obs, num_states = N, n_iters = 50)

            state_mean_speed = hmm.observations.params[0].T[1]
            index = np.argsort(state_mean_speed, -1)
                    
            #HMM.PlotTransition(transition_mat[index].T[index].T)
            np.save(title+"Transition_2.npy", transition_mat[index].T[index].T)

                    
            new_values = np.empty_like(states)
            for i, val in enumerate(index): new_values[states == val] = i
            states = new_values

                        
            #HMM.PlotStates(states, mouse_pos)
            np.save(title+"States_2.npy", states)
                    
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
            plt.savefig('/nfs/nhome/live/zimol/ProjectAeon/aeon_mecha/images/HMM/'+title+'Position_2.png')

            fig, axs = plt.subplots(5, 1, figsize = (10, 5*7-1))
            DATA = [SPEED, ACCE, VISIT_1, VISIT_2, CR]
            FEATURE = ['SPEED', 'ACCE', 'VISIT_1', 'VISIT_2', 'CR']
            for data, i in zip(DATA, range(len(DATA))):
                means = [np.mean(arr) for arr in data]
                std_devs = [np.std(arr) for arr in data]
                axs[i].bar(range(N), means, yerr=std_devs, capsize=5)
                axs[i].set_xticks(range(0, N), ['0', '1', '2', '3', '4', '5', '6']) # Change with number of states!!!
                axs[i].set_ylabel(FEATURE[i])
            plt.savefig('/nfs/nhome/live/zimol/ProjectAeon/aeon_mecha/images/HMM/'+title+'Data_2.png')
            

if __name__ == "__main__":
        main()