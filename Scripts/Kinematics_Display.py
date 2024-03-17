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
import Functions.patch as patch 

root = [Path("/ceph/aeon/aeon/data/raw/AEON2/experiment0.2")]

subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])
short_sessions = sessions.iloc[[4,16,17,20,23,24,25,26,28,29,30,31]]
long_sessions = sessions.iloc[[8, 10, 11, 14]]

def Display(session, title):
    start, end = session.enter, session.exit
    mouse_pos = api.load(root, exp02.CameraTop.Position, start=start, end=end)
        
    mouse_pos = kinematics.ProcessRawData(mouse_pos, root, start, end)
        
    # Add smoothed kinematics data to mouse_pos
    kinematics.AddKinematics(title, mouse_pos)
        
    #mouse_pos = mouse_pos.dropna(subset=['x'])
        
    # Draw plots
    fig, axs = plt.subplots(4,1, figsize = (40,24))
    mouse_pos.x.plot(ax = axs[0])
    mouse_pos.y.plot(ax = axs[0])
    mouse_pos.smoothed_position_x.plot(ax = axs[1])
    mouse_pos.smoothed_position_y.plot(ax = axs[1])
    mouse_pos.smoothed_speed.plot(ax = axs[2])
    mouse_pos.smoothed_acceleration.plot(ax = axs[3])
        
    axs[0].set_ylabel('Raw Pos.')
    axs[1].set_ylabel("Smoothed Pos.")
    axs[2].set_ylabel("Smoothed Vel.")
    axs[3].set_ylabel("Smoothed Acc.")
    plt.savefig('../Images/Kinematics/' + title+'.png')
    plt.show()

def DisplayShortSessions():
    for session, count in zip(list(short_sessions.itertuples()), range(len(short_sessions))):
        title = 'ShortSession'+str(count)
        Display(session, title)
        print(title)

def DisplayLongSessions():
    for session, count in zip(list(long_sessions.itertuples()), range(len(long_sessions))):
        title = 'LongSession'+str(count)
        Display(session, title)
        print(title)

def main():
    DisplayShortSessions()
    #DisplayLongSessions()


if __name__ == "__main__":
        main()