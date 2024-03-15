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


def PlotStates(states, mouse_pos, type, N, axs):
    color_names = ["blue","red","yellow", "green","brown","purple","orange"]
    colors = sns.xkcd_palette(color_names[0:N])
    cmap = gradient_cmap(colors)

    times = pd.to_datetime(mouse_pos.index)
    numerical_times = (times - times[0]).total_seconds().values
    states_array = states.reshape(1, -1)
    
    
    extent = [numerical_times[0], numerical_times[-1], 0, 1]
    cax = axs.imshow(states_array, aspect="auto", cmap=cmap, vmin=0, vmax=N-1, extent=extent)
    
    axs.set_xlabel('Time')
    axs.set_xticks(numerical_times[::len(numerical_times)//10])
    axs.set_xticklabels([time.strftime('%H:%M:%S') for time in times[::len(times)//10]], rotation=45, ha='right')
    
    axs.set_ylabel(type)
    axs.set_yticks([])

    return cax, axs
    

def main():
    root = [Path("/ceph/aeon/aeon/data/raw/AEON2/experiment0.2")]

    subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
    sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])
    short_sessions = sessions.iloc[[4,16,17,20,23,24,25,26,28,29,30,31]]
    long_sessions = sessions.iloc[[8, 10, 11, 14]]
    
    N = 5
    feature = ['weight', 'smoothed_speed', 'smoothed_acceleration']

    for session, j in zip(list(short_sessions.itertuples()), range(len(short_sessions))):
        title = 'ShortSession'+str(j)
        
        mouse_pos = pd.read_parquet('../Data/MousePos' + title + 'mousepos.parquet', engine='pyarrow')
        states = np.load('../Data/HMMStates'+title+"States_Unit.npy", allow_pickle = True)
        mouse_pos['states'] = pd.Series(states, index=mouse_pos.index)
        
        grouped = mouse_pos.groupby([pd.Grouper(freq='10S'), 'states']).size()
        prob = grouped.groupby(level=0).apply(lambda g: g / g.sum())
        states_prob = prob.unstack(level=-1).fillna(0)
        states_prob.index = states_prob.index.get_level_values(0)
        
        for i in range(N):
            if i not in states_prob.columns: states_prob[i] = 0
        
        x = np.array([np.array(states_prob[i].to_list()) for i in range(N)])

        states_ = []
        for i in range(len(x[0])):
            current_state = x.T[i]
            states_.append(np.argmax(current_state))
        

        fig, axs = plt.subplots(2, 1, figsize=(35, 8))
        cax, axs[0] = PlotStates(states, mouse_pos, type = 'States', N=5, axs = axs[0])
        cax, axs[1] = PlotStates(np.array(states_), states_prob, type = 'States Prob.', N=5, ax = axs[1])
        
        cbar = fig.colorbar(cax, ax=axs, orientation='vertical')
        cbar.set_ticks(np.arange(0, N))
        cbar.set_ticklabels([f'State {val}' for val in np.arange(0, N)])
        plt.savefig('../Images/HMM_States/' + title+'.png')

    
    
if __name__ == "__main__":
    main()