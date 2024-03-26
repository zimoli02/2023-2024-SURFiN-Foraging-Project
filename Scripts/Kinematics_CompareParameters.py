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

def CompareSession(session, title):
    start, end = session.enter, session.exit
    mouse_pos = api.load(root, exp02.CameraTop.Position, start=start, end=end)
    
    if title == 'ShortSession7': mouse_pos = kinematics.ProcessRawData(mouse_pos, root, start, end, exclude_maintenance=False)
    else: mouse_pos = kinematics.ProcessRawData(mouse_pos, root, start, end)
    obs = np.transpose(mouse_pos[["x", "y"]].to_numpy())
        
    fig, axs = plt.subplots(3,2, figsize = (50,24))
        
    # Manual parameters
    sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, V0, Z, R = kinematics.LDSParameters_Manual(dt=0.02)    
    Q = sigma_a**2*Qe

    filterRes_m = inference.filterLDS_SS_withMissingValues_np(
            y=obs, B=B, Q=Q, m0=m0, V0=V0, Z=Z, R=R)
    kinematics.AddKinematics_filter(mouse_pos, filterRes_m)
        
    mouse_pos.filtered_position_x.plot(ax = axs[0,0])
    mouse_pos.filtered_position_y.plot(ax = axs[0,0])
    mouse_pos.filtered_speed.plot(ax = axs[1,0])
    mouse_pos.filtered_acceleration.plot(ax = axs[2,0])
        
    # Learned parameters
    filterRes_l = np.load('../Data/ProcessedMouseKinematics/' + title+'filterRes.npz')
    kinematics.AddKinematics_filter(mouse_pos, filterRes_l)
    
    mouse_pos.filtered_position_x.plot(ax = axs[0,1])
    mouse_pos.filtered_position_y.plot(ax = axs[0,1])
    mouse_pos.filtered_speed.plot(ax = axs[1,1])
    mouse_pos.filtered_acceleration.plot(ax = axs[2,1])
        
        
    axs[0,0].set_ylabel('Filtered Pos.',fontsize = 16)
    axs[1,0].set_ylabel("Filtered Vel.",fontsize = 16)
    axs[2,0].set_ylabel("Filtered Acc.",fontsize = 16)
    
    axs[2,0].set_xlabel('Manual Para.',fontsize = 20)
    axs[2,1].set_xlabel("Learned Para.",fontsize = 20)
    plt.savefig('../Images/CompareParameters/' + title+'.png')
    plt.show()
    
    return filterRes_m['logLike'][0,0]/len(filterRes_m['xnn'][0][0]), filterRes_l['logLike'][0,0]/len(filterRes_l['xnn'][0][0])

def CompareShortSessions():
    LL_Manual, LL_Learned = [], []
    for session, count in zip(list(short_sessions.itertuples()), range(len(short_sessions))):
        title = 'ShortSession'+str(count)
        ll_manual, ll_learned = CompareSession(session, title)
        LL_Manual.append(ll_manual)
        LL_Learned.append(ll_learned)
        print(title)
    np.save('../Data/MouseKinematicParameters/ShortSessions_LL_Manual.npy', LL_Manual)
    np.save('../Data/MouseKinematicParameters/ShortSessions_LL_Learned.npy', LL_Learned)
    
def CompareLongSessions():
    LL_Manual, LL_Learned = [], []
    for session, count in zip(list(long_sessions.itertuples()), range(len(long_sessions))):
        title = 'ShortSession'+str(count)
        ll_manual, ll_learned = CompareSession(session, title)
        LL_Manual.append(ll_manual)
        LL_Learned.append(ll_learned)
        print(title)
    np.save('../Data/MouseKinematicParameters/LongSessions_LL_Manual.npy', LL_Manual)
    np.save('../Data/MouseKinematicParameters/LongSessions_LL_Learned.npy', LL_Learned)

def main():
    CompareShortSessions()
    #CompareLongSessions()


if __name__ == "__main__":
        main()