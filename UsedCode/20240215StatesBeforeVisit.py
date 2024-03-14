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

def VisitPatch(mouse_pos, patch = 'Patch1', period = 5):
    Visits = {'start':[],'end':[], 'duration':[], 'speed':[], 'acceleration':[], 'weight':[],'state':[]}
    
    groups = mouse_pos[patch].ne(mouse_pos[patch].shift()).cumsum()
    visits = mouse_pos[mouse_pos[patch] == 1].groupby(groups)[patch]
    for name, group in visits:
        Visits['start'].append(group.index[0])
        Visits['end'].append(group.index[-1])
        
        start_time = group.index[0]
        end_time = start_time - pd.Timedelta(seconds = period)
        if end_time < mouse_pos.index[0]: end_time = mouse_pos.index[0]
        
        pre_visit_data = mouse_pos.loc[end_time:start_time]
        
        pre_visit_data['states'].mean()
        
        Visits['duration'].append(mouse_pos.loc[group.index[0], patch + '_Leave_Time_Seconds'])
        Visits['speed'].append(pre_visit_data['smoothed_speed'].mean())
        Visits['acceleration'].append(pre_visit_data['smoothed_acceleration'].mean())
        Visits['weight'].append(pre_visit_data['weight'].mean())
        Visits['state'].append(pre_visit_data['states'].value_counts().idxmax())
    
    return pd.DataFrame(Visits)

def WheelDistance(Visits_Patch1, Visits_Patch2):
        Visits_Patch1 = Visits_Patch1.copy()
        Visits_Patch2 = Visits_Patch2.copy()

        Visits_Patch1['DistanceTravelledinVisit'] = 0
        for i in range(len(Visits_Patch1)):
                start, end = Visits_Patch1.start[i], Visits_Patch1.end[i]   
                encoder1 = api.load(root, exp02.Patch1.Encoder, start=start, end=end)
                if not encoder1.empty: 
                        w1 = -distancetravelled(encoder1.angle)
                        Visits_Patch1.loc[i, 'DistanceTravelledinVisit'] = w1[0]-w1[-1]
        
                
        Visits_Patch2['DistanceTravelledinVisit'] = 0
        for i in range(len(Visits_Patch2)):
                start, end = Visits_Patch2.start[i], Visits_Patch2.end[i]
                encoder2 = api.load(root, exp02.Patch2.Encoder, start=start, end=end)
                if not encoder2.empty:
                        w2 = -distancetravelled(encoder2.angle)
                        Visits_Patch2.loc[i, 'DistanceTravelledinVisit'] = w2[0]-w2[-1]


        return Visits_Patch1, Visits_Patch2

def main():
    
    Exp02Summary = api.load(root, exp02.Metadata).metadata[0].toDict()

    subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
    sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])
    short_sessions = sessions.iloc[[4,16,17,20,23,24,25,26,28,29,30,31]]
    
    N = 5
    feature = ['weight', 'smoothed_speed', 'smoothed_acceleration']
    
    for session, j in zip(list(short_sessions.itertuples()), range(len(short_sessions))):
        
        title = 'ShortSession'+str(j)
        print(title)
        mouse_pos = pd.read_parquet(title+'mousepos.parquet', engine='pyarrow')
        mouse_pos = mouse_pos[mouse_pos['smoothed_acceleration'] <= 60000]
        mouse_pos = DeleteRows(mouse_pos)

        states = np.load(title+"States.npy", allow_pickle = True)
    
        mouse_pos['states'] = pd.Series(states, index=mouse_pos.index)
        
        Visits_Patch1 = VisitPatch(mouse_pos, patch = 'Patch1')
        Visits_Patch2 = VisitPatch(mouse_pos, patch = 'Patch2')
        Visits_Patch1, Visits_Patch2 = WheelDistance(Visits_Patch1, Visits_Patch2) 
        
        Visits_Patch1_Distance = [np.array([0]) for _ in range(N)]
        Visits_Patch2_Distance = [np.array([0]) for _ in range(N)]
        Visits_Patches_Distance = [np.array([0]) for _ in range(N)]

        for i in range(N):
            v1, v2 = Visits_Patch1[Visits_Patch1['state'] == i].DistanceTravelledinVisit.to_numpy(), Visits_Patch2[Visits_Patch2['state'] == i].DistanceTravelledinVisit.to_numpy()
            
            Visits_Patch1_Distance[i] = np.concatenate((Visits_Patch1_Distance[i], v1))
            Visits_Patches_Distance[i] = np.concatenate((Visits_Patches_Distance[i], Visits_Patch1_Distance[i]))
            
            Visits_Patch2_Distance[i] = np.concatenate((Visits_Patch2_Distance[i], v2))
            Visits_Patches_Distance[i] = np.concatenate((Visits_Patches_Distance[i], Visits_Patch2_Distance[i]))
        
        fig, axs = plt.subplots(3,1,figsize = (9,17))
        axs[0].violinplot(Visits_Patch1_Distance)
        axs[0].set_ylabel("Patch1 Distance")
        axs[1].violinplot(Visits_Patch2_Distance)
        axs[1].set_ylabel("Patch2 Distance")
        axs[2].violinplot(Visits_Patches_Distance)
        axs[2].set_ylabel("Patch1&2 Distance")
        plt.savefig('/nfs/nhome/live/zimol/ProjectAeon/aeon_mecha/images/StateBeforeVisit/'+title+'.png')
            
if __name__ == "__main__":
        main()