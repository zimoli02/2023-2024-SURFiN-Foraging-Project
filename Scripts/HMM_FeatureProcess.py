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

import Functions.HMM as HMM
import Functions.kinematics as kinematics
import Functions.patch as patch


def main():
    root = [Path("/ceph/aeon/aeon/data/raw/AEON2/experiment0.2")]

    subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
    sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])
    short_sessions = sessions.iloc[[4,16,17,20,23,24,25,28,29,30,31]]
    long_sessions = sessions.iloc[[8, 10, 11, 14]]
    
    for session, i in zip(list(short_sessions.itertuples()), range(len(short_sessions))):
        title = 'ShortSession'+str(i)
        print(title)
        
        try:
            mouse_pos = pd.read_parquet('../Data/RawMouseKinematics/' + title + 'mousepos.parquet', engine='pyarrow')
        except FileNotFoundError:   
            start, end = session.enter, session.exit
            mouse_pos = api.load(root, exp02.CameraTop.Position, start=start, end=end)
            
            if i == 7: mouse_pos = kinematics.ProcessRawData(mouse_pos, root, start, end, exclude_maintenance=False)
            else: mouse_pos = kinematics.ProcessRawData(mouse_pos, root, start, end)
            
        start, end = mouse_pos.index[0], mouse_pos.index[-1]
                
        kinematics.AddKinematics(title, mouse_pos)    
            
        weight = api.load(root, exp02.Nest.WeightSubject, start=start, end=end)
        patch.AddWeight(mouse_pos, weight)
    
        patch.InPatch(mouse_pos, r = 30, interval = 5)
        
        mouse_pos = HMM.DeleteRows(mouse_pos, row = 5)
        mouse_pos.to_parquet('../Data/MousePos/' + title + 'mousepos.parquet', engine='pyarrow')
        print(title)
    
    
    

if __name__ == "__main__":
    main()