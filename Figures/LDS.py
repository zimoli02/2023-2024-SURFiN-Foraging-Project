import cv2
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
from pathlib import Path

import sys
from pathlib import Path

aeon_mecha_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(aeon_mecha_dir))

import aeon
import aeon.io.api as api
from aeon.schema.dataset import exp02
from aeon.analysis.utils import visits

import Functions.kinematics as kinematics
import Functions.inference as inference 

root = [Path("/ceph/aeon/aeon/data/raw/AEON2/experiment0.2")]

subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])
short_sessions = sessions.iloc[[4,16,17,20,23,24,25,26,28,29,30,31]]
long_sessions = sessions.iloc[[8, 10, 11, 14]]

start_, end_ = 10*60*50, 11*60*50

def GetRawData():
    start, end = pd.Timestamp('2022-03-15 12:40:36.282139778'), pd.Timestamp('2022-03-15 15:56:55.801119804')
    mouse_pos = api.load(root, exp02.CameraTop.Position, start=start, end=end)
    mouse_pos = kinematics.ProcessRawData(mouse_pos, root, start, end, fix_nan=False, fix_nest=False)
    
    obs = np.transpose(mouse_pos[["x", "y"]].to_numpy())[:, start_:end_]
    np.save('../Data/MouseKinematicParameters/ShortSession0Raw.npy', obs)
    
    return obs

def main():
    # Display short session 0 as an example
    try:
        obs = np.load('../Data/MouseKinematicParameters/ShortSession0Raw.npy', allow_pickle=True)
    except FileNotFoundError:
        obs = GetRawData()
    

    start, end = 2200, 2260
    
    x, y = obs[0][start:end], obs[1][start:end]
        
    smoothRes = np.load('../Data/ProcessedMouseKinematics/ShortSession0smoothRes.npz')
    smooth_x = smoothRes['xnN'][0][0][start_:end_][start:end]
    smooth_y = smoothRes['xnN'][3][0][start_:end_][start:end]
    smooth_x_var = smoothRes['VnN'][0][0][start_:end_][start:end]
    smooth_y_var = smoothRes['VnN'][3][3][start_:end_][start:end]

    time = np.arange(0, len(x), 1)
        
    fig, axs = plt.subplots(1,2, figsize = (12,4))
    axs[0].plot(time, x, color = 'black', linewidth = 0.5, label = 'Raw X')
    axs[0].scatter(time, x, color = 'black', s = 2)
    axs[0].plot(time, smooth_x, color = 'red', linewidth = 0.5, label = 'Smoothed X')
    axs[0].scatter(time, smooth_x, color = 'red', s = 2)
    axs[0].fill_between(time, smooth_x - 1.65*(smooth_x_var**0.5), smooth_x + 1.65*(smooth_x_var**0.5), color = 'pink', alpha = 0.7)
    axs[0].legend(loc = 'upper right')

    axs[1].plot(time, y, color = 'black', linewidth = 0.5, label = 'Raw Y')
    axs[1].scatter(time, y, color = 'black', s = 2)
    axs[1].plot(time, smooth_y, color = 'red', linewidth = 0.5, label = 'Smoothed Y')
    axs[1].scatter(time, smooth_y, color = 'red', s=2)
    axs[1].fill_between(time, smooth_y - 1.65*(smooth_y_var**0.5), smooth_y + 1.65*(smooth_y_var**0.5), color = 'pink', alpha = 0.7)
    axs[1].legend(loc = 'upper right')

    axs[0].set_ylabel('Position X')
    axs[0].set_yticks([584, 586, 588, 590, 592, 594, 596])
    axs[1].set_ylabel('Position Y')
    axs[1].set_yticks([230, 235, 240, 245, 250])
        
    axs[0].set_xticks([])
    axs[1].set_xticks([])

    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
        
    plt.autoscale()
    plt.savefig('../Figures/LDS.png')
    plt.show()

if __name__ == "__main__":
        main()