import cv2
import matplotlib.pyplot as plt
from dotmap import DotMap

import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import statsmodels.api as sm
import scipy.stats as stats

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib.animation import FuncAnimation, FFMpegWriter

import os
import subprocess

import sys
from pathlib import Path
current_script_path = Path(__file__).resolve()

functions_dir = current_script_path.parents[1] / 'Functions'
sys.path.insert(0, str(functions_dir))
import mouse as mouse

aeon_mecha_dir = current_script_path.parents[2] / 'aeon_mecha' 
sys.path.insert(0, str(aeon_mecha_dir))
import aeon
import aeon.io.api as api
from aeon.io import reader, video
from aeon.schema.schemas import social02
from aeon.analysis.utils import visits, distancetravelled

LABELS = [
    ['AEON3', 'Pre','BAA-1104045'],
    ['AEON3', 'Pre','BAA-1104047'],
    ['AEON3', 'Post','BAA-1104045'],
    ['AEON3', 'Post','BAA-1104047'],
    ['AEON4', 'Pre','BAA-1104048'], 
    ['AEON4', 'Pre','BAA-1104049'],
    ['AEON4', 'Post','BAA-1104048'],
    ['AEON4', 'Post','BAA-1104049']
]


def main():
    
    '''for label in LABELS:
        aeon_exp_name, type_name, mouse_name = label[0], label[1], label[2]
        print('Start Processing: ', type_name, "-", mouse_name)
        
        Mouse = mouse.Mouse(aeon_exp = aeon_exp_name, type = type_name, mouse = mouse_name)
        for i in range(len(Mouse.starts)):
            print('Session '+str(i+1), Mouse.starts[i], Mouse.ends[i])
        for i in range(len(Mouse.experiment_starts)):
            print('Fix_Session '+str(i+1), Mouse.experiment_starts[i], Mouse.experiment_ends[i])'''

    label = LABELS[2]
    aeon_exp_name, type_name, mouse_name = label[0], label[1], label[2]
    Mouse = mouse.Mouse(aeon_exp = aeon_exp_name, type = type_name, mouse = mouse_name)
    '''video_metadata = aeon.load(Mouse.root, social02.CameraTop.Video, start=Mouse.mouse_pos.index[0]+ pd.Timedelta('30S'), end=Mouse.mouse_pos.index[0] + pd.Timedelta('60S'))
    
    video_metadata.index = video_metadata.index.round("20L")  # round timestamps to nearest 20 ms
    frames = video.frames(video_metadata)  # get actual frames based on vid metadata
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid = cv2.VideoWriter("../Videos/test.mp4", fourcc=fourcc, fps=50, frameSize=(1440, 1080))  # will save to current dir
    for f in frames:  # write out frames to vid, frame-by-frame
        vid.write(f)
    vid.release()'''
    plt.scatter(Mouse.mouse_pos.smoothed_position_x, Mouse.mouse_pos.smoothed_position_y)
    plt.savefig('test.png')
    
        
    
if __name__ == "__main__":
    main()