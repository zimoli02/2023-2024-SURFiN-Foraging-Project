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

root = [Path("/ceph/aeon/aeon/data/raw/AEON2/experiment0.2")]

subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])
short_sessions = sessions.iloc[[4,16,17,20,23,24,25,28,29,30,31]]
long_sessions = sessions.iloc[[8, 10, 11, 14]]

def Kinematics(title, mouse_pos):
    kinematics.AddKinematics(title, mouse_pos) 
    return mouse_pos 

def Weight(mouse_pos):
    start, end = mouse_pos.index[0], mouse_pos.index[-1]
                
    weight = api.load(root, exp02.Nest.WeightSubject, start=start, end=end)
    patch.AddWeight(mouse_pos, weight)
    
    return mouse_pos

def InPatch(mouse_pos, r = 30, interval = 5):
    patch.InPatch(mouse_pos, r = r, interval = interval)
    return mouse_pos

def InBothPatches(mouse_pos, r = 100):
    mouse_pos = patch.PositionInPatch(mouse_pos, r = r) 
    return mouse_pos

def DistanceToOri(mouse_pos):
    mouse_pos = patch.Radius(mouse_pos)
    return mouse_pos

def PositionInArena(mouse_pos):
    return patch.PositionInArena(mouse_pos)

def main():
    Kinematics_Update = False
    Weight_Update = False
    
    Patch1_Update = False
    Patch_Update = False
    r_Update = False
    Arena_Update = False
    
    for session, i in zip(list(short_sessions.itertuples()), range(len(short_sessions))):
        title = 'ShortSession'+str(i)

        try:
            mouse_pos = pd.read_parquet('../Data/MousePos/' + title + 'mousepos.parquet', engine='pyarrow')
        except FileNotFoundError:  
            try:
                mouse_pos = pd.read_parquet('../Data/RawMouseKinematics/' + title + 'mousepos.parquet', engine='pyarrow')
            except FileNotFoundError:   
                start, end = session.enter, session.exit
                mouse_pos = api.load(root, exp02.CameraTop.Position, start=start, end=end)
                mouse_pos = kinematics.ProcessRawData(mouse_pos, root, start, end)

        if 'smoothed_speed' not in mouse_pos.columns or Kinematics_Update:
            mouse_pos = Kinematics(title, mouse_pos)    
        
        if 'weight' not in mouse_pos.columns or Weight_Update:
            mouse_pos = Weight(mouse_pos)
        
        if 'Patch1' not in mouse_pos.columns or Patch1_Update:
            mouse_pos = InPatch(mouse_pos, r = 30, interval = 5)
        
        if 'Patch' not in mouse_pos.columns or Patch_Update:
            mouse_pos = InBothPatches(mouse_pos, r = 100)  
        
        if 'r' not in mouse_pos.columns or r_Update:
            mouse_pos = DistanceToOri(mouse_pos)
            
        if 'Arena' not in mouse_pos.columns or Arena_Update:
            mouse_pos = PositionInArena(mouse_pos)
        
        if (mouse_pos.index[1] - mouse_pos.index[0]) < pd.Timedelta('0.04S'):
            mouse_pos = HMM.DeleteRows(mouse_pos, row = 5)
            
        mouse_pos.to_parquet('../Data/MousePos/' + title + 'mousepos.parquet', engine='pyarrow')
        print(title)
    
    
    

if __name__ == "__main__":
    main()