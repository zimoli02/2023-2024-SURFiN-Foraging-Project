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
import Functions.patch as patch


root = [Path("/ceph/aeon/aeon/data/raw/AEON2/experiment0.2")]
subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])
short_sessions = sessions.iloc[[4,16,17,20,23,24,25,26,28,29,30,31]]
long_sessions = sessions.iloc[[8, 10, 11, 14]]

    
    
def ProcessSession(session, title, param):
    try:
        mouse_pos = pd.read_parquet('../Data/RawMouseKinematics/' + title + 'mousepos.parquet', engine='pyarrow')
    except FileNotFoundError:   
        start, end = session.enter, session.exit
        mouse_pos = api.load(root, exp02.CameraTop.Position, start=start, end=end)
        
        if title == 'ShortSession7': maintenance = False
        else: maintenance = True
        
        if title[:12] == 'ShortSession':
            mouse_pos = kinematics.ProcessRawData(mouse_pos, root, start, end, exclude_maintenance=maintenance)
        else:
            mouse_pos = kinematics.ProcessRawData(mouse_pos, root, start, end, exclude_maintenance=maintenance, fix_nan=False, fix_nest=False)
            mouse_pos_subs = patch.SeparateDF(mouse_pos)
            dfs = []
            for mouse_pos_sub in mouse_pos_subs:      
                mouse_pos_sub = kinematics.FixNan(mouse_pos_sub)
                dfs.append(mouse_pos_sub)
            mouse_pos = dfs[0]
            for df in dfs[1:]: mouse_pos = mouse_pos.add(df, fill_value=0)

    obs = np.transpose(mouse_pos[["x", "y"]].to_numpy())
    

    if param != 'Manual':
        P = np.load('../Data/MouseKinematicParameters/' + title + 'Parameters.npz', allow_pickle=True)
    else:
        P = np.load('../Data/MouseKinematicParameters/ManualParameters.npz', allow_pickle=True)
    
    sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, V0, Z, R = P['sigma_a'].item(), P['sigma_x'].item(), P['sigma_y'].item(), P['sqrt_diag_V0_value'].item(), P['B'], P['Qe'], P['m0'], P['V0'], P['Z'], P['R']
        
    Q = (sigma_a**2) * Qe

    # Filtering
    filterRes = inference.filterLDS_SS_withMissingValues_np(
        y=obs, B=B, Q=Q, m0=m0, V0=V0, Z=Z, R=R)
    np.savez_compressed('../Data/ProcessedMouseKinematics/' + title + 'filterRes.npz', **filterRes)
        
    # Smoothing
    smoothRes = inference.smoothLDS_SS( 
        B=B, xnn=filterRes["xnn"], Vnn=filterRes["Vnn"],
        xnn1=filterRes["xnn1"], Vnn1=filterRes["Vnn1"], m0=m0, V0=V0)
    np.savez_compressed('../Data/ProcessedMouseKinematics/' + title + 'smoothRes.npz', **smoothRes)
    
def ProcessShortSessions(param):
    for session, count in zip(list(short_sessions.itertuples()), range(len(short_sessions))):
        title = 'ShortSession'+str(count)
        ProcessSession(session, title, param = param)
        print(title)

def ProcessLongSessions(param):
    for session, count in zip(list(long_sessions.itertuples()), range(len(long_sessions))):
        if count < 2 : continue
        title = 'LongSession'+str(count)
        ProcessSession(session, title, param = param)
        print(title)
        
def main():
        
    ProcessShortSessions(param = 'Learned')
    #ProcessLongSessions(param = 'Learned')
        

if __name__ == "__main__":
        main()