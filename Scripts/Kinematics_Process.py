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


import Functions.inference as inference
import Functions.kinematics as kinematics

def main():
    root = [Path("/ceph/aeon/aeon/data/raw/AEON2/experiment0.2")]

    subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
    sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])
    short_sessions = sessions.iloc[[4,16,17,20,23,24,25,26,28,29,30,31]]
    long_sessions = sessions.iloc[[8, 10, 11, 14]]
    
    sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, V0, Z, R = kinematics.LDSParameters_Manual(dt=0.02)
    dt = 0.019968
    
    for session, count in zip(list(short_sessions.itertuples()), range(len(short_sessions))):
        title = 'ShortSession'+str(count)
        start, end = session.enter, session.exit
        mouse_pos = api.load(root, exp02.CameraTop.Position, start=start, end=end)
        
        mouse_pos = kinematics.ProcessRawData(mouse_pos, root, start, end)

        obs = np.transpose(mouse_pos[["x", "y"]].to_numpy())

        # Use the first 20 seconds position data to learn the LDS parameters
        sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, V0, Z, R = kinematics.LDSParameters_Learned(obs[:,:6000], dt, sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, Z)
        np.savez('../Data/MouseKinematicParameters/' + title + 'Parameters.npz', sigma_a = sigma_a, sigma_x = sigma_x, sigma_y = sigma_y, sqrt_diag_V0_value = sqrt_diag_V0_value, B = B, Qe = Qe, m0 = m0, V0 = V0, Z = Z, R = R)
        
        Q = sigma_a**2*Qe

        # Filtering
        filterRes = inference.filterLDS_SS_withMissingValues_np(
            y=obs, B=B, Q=Q, m0=m0, V0=V0, Z=Z, R=R)
        np.savez_compressed('../Data/ProcessedMouseKinematics/' + title + 'filterRes.npz', **filterRes)
        
        # Smoothing
        smoothRes = inference.smoothLDS_SS( 
            B=B, xnn=filterRes["xnn"], Vnn=filterRes["Vnn"],
            xnn1=filterRes["xnn1"], Vnn1=filterRes["Vnn1"], m0=m0, V0=V0)
        np.savez_compressed('../Data/ProcessedMouseKinematics/' + title + 'smoothRes.npz', **smoothRes)
    
    for session, count in zip(list(long_sessions.itertuples()), range(len(long_sessions))):
        title = 'LongSession'+str(count)
        start, end = session.enter, session.exit
        mouse_pos = api.load(root, exp02.CameraTop.Position, start=start, end=end)
        
        mouse_pos = kinematics.ProcessRawData(mouse_pos, root, start, end)
        
        obs = np.transpose(mouse_pos[["x", "y"]].to_numpy())

        
        # Use the first 20 seconds position data to learn the LDS parameters
        sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, V0, Z, R = kinematics.LDSParameters_Learned(obs[:,:6000], dt, sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, Z)
        np.savez('../Data/MouseKinematicParameters/' + title + 'Parameters.npz', sigma_a = sigma_a, sigma_x = sigma_x, sigma_y = sigma_y, sqrt_diag_V0_value = sqrt_diag_V0_value, B = B, Qe = Qe, m0 = m0, V0 = V0, Z = Z, R = R)
        
        Q = sigma_a**2*Qe

        # Filtering
        filterRes = inference.filterLDS_SS_withMissingValues_np(
            y=obs, B=B, Q=Q, m0=m0, V0=V0, Z=Z, R=R)
        np.savez_compressed('../Data/ProcessedMouseKinematics/' + title + 'filterRes.npz', **filterRes)
        
        # Smoothing
        smoothRes = inference.smoothLDS_SS( 
            B=B, xnn=filterRes["xnn"], Vnn=filterRes["Vnn"],
            xnn1=filterRes["xnn1"], Vnn1=filterRes["Vnn1"], m0=m0, V0=V0)
        np.savez_compressed('../Data/ProcessedMouseKinematics/' + title + 'smoothRes.npz', **smoothRes)        

if __name__ == "__main__":
        main()