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


root = [Path("/ceph/aeon/aeon/data/raw/AEON2/experiment0.2")]
subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])

short_sessions = sessions.iloc[[4,16,17,20,23,24,25,28,29,30,31]]
long_sessions = sessions.iloc[[8, 10, 11, 14]]

def ProcessSession(session, title):
    try:
        P = np.load('../Data/MouseKinematicParameters/' + title + 'Parameters.npz', allow_pickle=True)
        obs = np.load('../Data/MouseKinematicParameters/' + title + 'OBS.npy', allow_pickle=True)
    except FileNotFoundError:
        try:
            mouse_pos = pd.read_parquet('../Data/RawMouseKinematics/' + title + 'mousepos.parquet', engine='pyarrow')
        except FileNotFoundError:
            start, end = session.enter, session.exit
            
            mouse_pos = api.load(root, exp02.CameraTop.Position, start=start, end=end)
            mouse_pos = kinematics.ProcessRawData(mouse_pos, root, start, end, exclude_maintenance=True, fix_nan=False, fix_nest=False)
            
            if title[0:11] == 'LongSession': 
                end = start+pd.Timedelta('2H')
                latest_valid_index = mouse_pos.loc[start:end, 'x'].last_valid_index()
                end = latest_valid_index+pd.Timedelta('0.01S')
            mouse_pos = kinematics.FixNan(mouse_pos[start:end])
        
        obs = np.transpose(mouse_pos[["x", "y"]].to_numpy())[:, :10*60*50]
        np.save('../Data/MouseKinematicParameters/' + title + 'OBS.npy', obs)
        
        P = np.load('../Data/MouseKinematicParameters/ManualParameters.npz', allow_pickle=True)
        sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, V0, Z, R = P['sigma_a'].item(), P['sigma_x'].item(), P['sigma_y'].item(), P['sqrt_diag_V0_value'].item(), P['B'], P['Qe'], P['m0'], P['V0'], P['Z'], P['R']

        #First 10 min of the data
        sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, m0, V0, Z, R = kinematics.LDSParameters_Learned(obs, sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, Z)
        np.savez('../Data/MouseKinematicParameters/' + title + 'Parameters.npz', sigma_a = sigma_a, sigma_x = sigma_x, sigma_y = sigma_y, sqrt_diag_V0_value = sqrt_diag_V0_value, B = B, Qe = Qe, m0 = m0, V0 = V0, Z = Z, R = R)
        print('parameters')
    
def ProcessShortSessions(param = 'Learned'):
    for session, count in zip(list(short_sessions.itertuples()), range(len(short_sessions))):
        title = 'ShortSession'+str(count)
        ProcessSession(session, title)
        print(title)

def ProcessLongSessions(param = 'Learned'):
    for session, count in zip(list(long_sessions.itertuples()), range(len(long_sessions))):
        title = 'LongSession'+str(count)
        ProcessSession(session, title)
        print(title)
        
def main():
    
    sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, V0, Z, R = kinematics.LDSParameters_Manual(dt=0.02)
    np.savez('../Data/MouseKinematicParameters/ManualParameters.npz', sigma_a = sigma_a, sigma_x = sigma_x, sigma_y = sigma_y, sqrt_diag_V0_value = sqrt_diag_V0_value, B = B, Qe = Qe, m0 = m0, V0 = V0, Z = Z, R = R)


        
    ProcessShortSessions()
    ProcessLongSessions()
        

if __name__ == "__main__":
        main()