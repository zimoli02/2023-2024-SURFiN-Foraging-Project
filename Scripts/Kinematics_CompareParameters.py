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

def main():
    root = [Path("/ceph/aeon/aeon/data/raw/AEON2/experiment0.2")]

    subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
    sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])
    short_sessions = sessions.iloc[[4,16,17,20,23,24,25,26,28,29,30,31]]
    long_sessions = sessions.iloc[[8, 10, 11, 14]]
    
    LL_Manual, LL_Learned = [], []
    for session, count in zip(list(short_sessions.itertuples()), range(len(short_sessions))):
        title = 'ShortSession'+str(count)
        print(title)
        
        start, end = session.enter, session.exit
        mouse_pos = api.load(root, exp02.CameraTop.Position, start=start, end=end)
        
        mouse_pos = kinematics.ProcessRawData(mouse_pos, root, start, end)
        obs = np.transpose(mouse_pos[["x", "y"]].to_numpy())
        
        fig, axs = plt.subplots(3,2, figsize = (50,24))
        
        # Manual parameters
        sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, V0, Z, R = kinematics.LDSParameters_Manual(dt=0.02)    
        Q = sigma_a**2*Qe

        filterRes_m = inference.filterLDS_SS_withMissingValues_np(
            y=obs, B=B, Q=Q, m0=m0, V0=V0, Z=Z, R=R)
        LL_Manual.append(filterRes_m['logLike'])
        
        kinematics.AddKinematics_filter(mouse_pos, filterRes_m)
        
        mouse_pos.filtered_position_x.plot(ax = axs[0,0])
        mouse_pos.filtered_position_y.plot(ax = axs[0,0])
        mouse_pos.filtered_speed.plot(ax = axs[1,0])
        mouse_pos.filtered_acceleration.plot(ax = axs[2,0])
        
        # Learned parameters
        filterRes_l = np.load('../Data/ProcessedMouseKinematics/' + title+'filterRes.npz')
        
        LL_Learned.append(filterRes_l['logLike'])

        kinematics.AddKinematics_filter(mouse_pos, filterRes_l)
    
        mouse_pos.filtered_position_x.plot(ax = axs[0,1])
        mouse_pos.filtered_position_y.plot(ax = axs[0,1])
        mouse_pos.filtered_speed.plot(ax = axs[1,1])
        mouse_pos.filtered_acceleration.plot(ax = axs[2,1])
        
        
        axs[0,0].set_ylabel('Filtered Pos.')
        axs[1,0].set_ylabel("Filtered Vel.")
        axs[2,0].set_ylabel("Filtered Acc.")
        axs[2,0].set_xlabel('Manual Para.')
        axs[2,1].set_ylabel("Learned Para.")
        plt.savefig('../Images/CompareParameters/' + title+'.png')
        plt.show()
        
        np.save('../Data/MouseKinematicParameters/LL_Manual.npy', LL_Manual)
        np.save('../Data/MouseKinematicParameters/LL_Learned.npy', LL_Learned)


if __name__ == "__main__":
        main()