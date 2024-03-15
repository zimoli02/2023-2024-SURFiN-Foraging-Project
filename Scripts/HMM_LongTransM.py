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

from SSM.ssm.plots import gradient_cmap

import Functions.HMM as HMM
import Functions.kinematics as kinematics
import Functions.patch as patch

root = [Path("/ceph/aeon/aeon/data/raw/AEON2/experiment0.2")]

subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])
short_sessions = sessions.iloc[[4,16,17,20,23,24,25,26,28,29,30,31]]
long_sessions = sessions.iloc[[8, 10, 11, 14]]

feature = ['weight', 'smoothed_speed', 'smoothed_acceleration']

def Days(id):
    if id == 0:
        START = ['2022-03-30 07:00:00','2022-04-01 07:00:00','2022-04-02 07:00:00','2022-04-03 07:00:00']
        END = ['2022-03-31 07:00:00','2022-04-02 07:00:00','2022-04-03 07:00:00','2022-04-04 07:00:00']
    #if id == 1: 
    else:
        START, END = [],[]
    return START, END
        

def main():
    N = 8
    for session, j in zip(list(long_sessions.itertuples()), range(len(long_sessions))):
        title = 'LongSession' + str(j) 
        START, END = Days(j)
        if len(START) == 0: continue
        
        start, end = session.enter, session.exit
        mouse = api.load(root, exp02.CameraTop.Position, start=start, end=end)     
        mouse = kinematics.ProcessRawData(mouse, root, start, end)    
        patch.AddKinematics(title, mouse) 
        
        mouse_pos_subs = patch.SeparateDF(mouse)
        dfs = []
        for mouse_pos_sub in mouse_pos_subs:
            mouse_pos_sub = mouse_pos_sub[mouse_pos_sub['smoothed_speed'] <= 2000]
            mouse_pos_sub = mouse_pos_sub[mouse_pos_sub['smoothed_acceleration'] <= 60000]
            mouse_pos_sub = patch.DeleteRows(mouse_pos_sub)
            mouse_pos_sub = mouse_pos_sub.loc[:,['x']]
            dfs.append(mouse_pos_sub)
        mouse_pos = dfs[0]
        for df in dfs[1:]: mouse_pos = mouse_pos.add(df, fill_value=0)
        
        states = np.load('../Data/HMMStates'+title+"States_Unit.npy", allow_pickle = True)
        mouse_pos['states'] = pd.Series(states, index=mouse_pos.index)
        
        grouped = mouse_pos.groupby([pd.Grouper(freq='10S'), 'states']).size()
        prob = grouped.groupby(level=0).apply(lambda g: g / g.sum())
        states_prob = prob.unstack(level=-1).fillna(0)
        states_prob.index = states_prob.index.get_level_values(0)
        
        fig, axs = plt.subplots(1,len(START), figsize=(len(START)*4, 4))
        for i in range(len(START)):
            start, end = pd.Timestamp(START[i]),pd.Timestamp(END[i])
            period_data = states_prob.loc[start:end]
            states_ = period_data['states'].to_numpy()
            
            manual_trans_mat = np.zeros((N,N))
            for k in range(1,len(states_)): manual_trans_mat[states_[k-1]][states_[k]] += 1
            for k in range(N): 
                if np.sum(manual_trans_mat[k]) != 0: manual_trans_mat[k] = manual_trans_mat[k]/np.sum(manual_trans_mat[k])

            annot_array = np.array([[round(item, 2) for item in row] for row in manual_trans_mat])
            
            sns.heatmap(manual_trans_mat, cmap='YlGnBu', ax = axs[i], square = 'True', cbar = False, annot=annot_array, annot_kws = {'size':10})
        fig.suptitle("Manual-Calculated Transition Matrix")
        plt.savefig('../Images/HMM_TransitionM/' + title +'MultipleDays.png')
        
    

if __name__ == "__main__":
    main()