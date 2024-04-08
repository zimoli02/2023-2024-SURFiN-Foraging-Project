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
short_sessions = sessions.iloc[[4,16,17,20,23,24,25,28,29,30,31]]
long_sessions = sessions.iloc[[8, 10, 11, 14]]

start_, end_ = 10*60*50, 11*60*50

scale = 2e-3

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
    

    start, end = 2220, 2240
    dt = 0.02

    
    x = obs[0][start:end] * scale
    x_vel = np.array([(x[i+1]-x[i])/dt for i in range(len(x)-1)])
    x_vel = np.concatenate((np.array([0]), x_vel))
    x_acce = np.array([(x_vel[i+1]-x_vel[i])/dt for i in range(len(x_vel)-1)])
    x_acce = np.concatenate((np.array([0]), x_acce))
        
    smoothRes = np.load('../Data/ProcessedMouseKinematics/ShortSession0smoothRes.npz')
    smooth_x = smoothRes['xnN'][0][0][start_:end_][start:end] * scale
    smooth_x_var = smoothRes['VnN'][0][0][start_:end_][start:end] * scale ** 2

    smooth_x_vel = smoothRes['xnN'][1][0][start_:end_][start:end] * scale
    smooth_x_vel_var = smoothRes['VnN'][1][1][start_:end_][start:end] * scale ** 2
    
    smooth_x_acce = smoothRes['xnN'][2][0][start_:end_][start:end] * scale
    smooth_x_acce_var = smoothRes['VnN'][2][2][start_:end_][start:end] * scale ** 2

    time = np.arange(0, len(x), 1)
    time = time * dt
        
    fig, axs = plt.subplots(1,3, figsize = (16,8))
    axs[0].plot(time, x, color = 'black', linewidth = 1, label = 'Raw')
    axs[0].scatter(time, x, color = 'black', s = 6)
    axs[0].plot(time, smooth_x, color = 'red', linewidth = 1, label = 'Smoothed')
    axs[0].scatter(time, smooth_x, color = 'red', s = 6)
    axs[0].fill_between(time, smooth_x - 1.65*(smooth_x_var**0.5), smooth_x + 1.65*(smooth_x_var**0.5), color = 'pink', alpha = 0.7, label = '95% C.I.')
    axs[0].legend(loc = 'upper right')

    axs[1].plot(time, x_vel, color = 'black', linewidth = 1, label = 'Raw')
    axs[1].scatter(time, x_vel, color = 'black', s = 6)
    axs[1].plot(time, smooth_x_vel, color = 'blue', linewidth = 1, label = 'Smoothed')
    axs[1].scatter(time, smooth_x_vel, color = 'blue', s=6)
    axs[1].fill_between(time, smooth_x_vel - 1.65*(smooth_x_vel_var**0.5), smooth_x_vel + 1.65*(smooth_x_vel_var**0.5), color = 'lightblue', alpha = 0.7, label = '95% C.I.')
    axs[1].legend(loc = 'lower right')
    
    axs[2].plot(time, x_acce, color = 'black', linewidth = 1, label = 'Raw')
    axs[2].scatter(time, x_acce, color = 'black', s = 6)
    axs[2].plot(time, smooth_x_acce, color = 'green', linewidth = 1, label = 'Smoothed')
    axs[2].scatter(time, smooth_x_acce, color = 'green', s=6)
    axs[2].fill_between(time, smooth_x_acce - 1.65*(smooth_x_acce_var**0.5), smooth_x_acce + 1.65*(smooth_x_acce_var**0.5), color = 'lightgreen', alpha = 0.7, label = '95% C.I.')
    axs[2].legend(loc = 'upper right')

    

    axs[0].set_ylabel('Position (m)', fontsize = 16)
    axs[1].set_ylabel('Speed (m/s)', fontsize = 16)
    axs[2].set_ylabel('Acceleration (m/s$^2$)', fontsize = 16)
    
    '''
    axs[0].set_title('Position', fontsize = 20)
    axs[1].set_title('Speed', fontsize = 20)
    axs[2].set_title('Acceleration', fontsize = 20)
    '''

    for i in range(3):
        axs[i].set_xticks(np.arange(0,time[-1]+0.01, 0.05))
        axs[i].set_xlabel('Time (s)', fontsize = 16)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        
        
    plt.tight_layout()
    plt.savefig('../Figures/Results/LDS.png')
    plt.show()

if __name__ == "__main__":
        main()