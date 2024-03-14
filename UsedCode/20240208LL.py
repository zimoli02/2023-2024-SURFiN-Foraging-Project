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
    Patch_Pos = [device['Regions']['ArrayOfPoint'] for device in Exp02Summary['Devices'] if device['Name'].startswith('ActivityPatch')]

    subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
    sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])
    
    sessions_ = sessions.iloc[[4,8,10,11,14,16,17,20,23,24,25,26,28,29,30,31]]
    short_sessions = sessions.iloc[[4,16,17,20,23,24,25,26,28,29,30,31]]
    long_sessions = sessions.iloc[[8, 10, 11, 14]]

    S = []

    for n in range(3,10,1):
        N = n
        
        feature = ['weight', 'smoothed_speed', 'smoothed_acceleration']
        LL = []

        for k in range(len(short_sessions)):
            title = 'ShortSession'+str(k)
            
            mouse_pos = pd.read_parquet(title+'mousepos.parquet', engine='pyarrow')
            mouse_pos = mouse_pos[mouse_pos['smoothed_acceleration'] <= 60000]
            mouse_pos = DeleteRows(mouse_pos)
                    
            obs = np.array(mouse_pos[feature])
            hmm, states, transition_mat, lls = HMM.FitHMM(obs, num_states = N, n_iters = 50)
            ll = []

            for session, j in zip(list(short_sessions.itertuples()), range(len(short_sessions))):
                title = 'ShortSession'+str(j)
                mouse_pos = pd.read_parquet(title+'mousepos.parquet', engine='pyarrow')
                mouse_pos = mouse_pos[mouse_pos['smoothed_acceleration'] <= 60000]
                mouse_pos = DeleteRows(mouse_pos)
                    
                obs = np.array(mouse_pos[feature])
                states = hmm.most_likely_states(obs)
                    
                ll.append(hmm.log_likelihood(obs))
                
            LL.append(np.mean(ll))

        S.append(LL)
    
    np.save("SumOfLogLikelihood.npy", S)

if __name__ == "__main__":
        main()