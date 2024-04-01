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

short_sessions = sessions.iloc[[4,16,17,20,23,24,25,26,28,29,30,31]]
long_sessions = sessions.iloc[[8, 10, 11, 14]]

def ProcessSession(session, title):
    try:
        mouse_pos = pd.read_parquet('../Data/RawMouseKinematics/' + title + 'mousepos.parquet', engine='pyarrow')
    except FileNotFoundError:
        start, end = session.enter, session.exit
        
        if title == 'ShortSession7': maintenance = False
        else: maintenance = True
        
        mouse_pos = api.load(root, exp02.CameraTop.Position, start=start, end=end)
        mouse_pos = mouse_pos[['x','y']]
        mouse_pos = kinematics.ProcessRawData(mouse_pos, root, start, end, exclude_maintenance=maintenance, fix_nan=True, fix_nest=False)
        
        
        mouse_pos.to_parquet('../Data/RawMouseKinematics/' + title + 'mousepos.parquet', engine='pyarrow')
    
def ProcessShortSessions():
    for session, count in zip(list(short_sessions.itertuples()), range(len(short_sessions))):
        title = 'ShortSession'+str(count)
        ProcessSession(session, title)
        print(title)

def ProcessLongSessions():
    for session, count in zip(list(long_sessions.itertuples()), range(len(long_sessions))):
        if count != 0: continue
        title = 'LongSession'+str(count)
        ProcessSession(session, title)
        print(title)
        
def main():
    

    #ProcessShortSessions()
    ProcessLongSessions()
        

if __name__ == "__main__":
        main()