import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

import sys
from pathlib import Path

aeon_mecha_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(aeon_mecha_dir))

import aeon
import aeon.io.api as api
from aeon.schema.dataset import exp02
from aeon.analysis.utils import visits

import Functions.patch as patch

root = [Path("/ceph/aeon/aeon/data/raw/AEON2/experiment0.2")]

subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])
short_sessions = sessions.iloc[[4,16,17,20,23,24,25,28,29,30,31]]
long_sessions = sessions.iloc[[8, 10, 11, 14]]

def ConcatenateSessions():
    dfs = []
    for i in range(len(short_sessions)):
        title = 'ShortSession'+str(i)
        Visits_Patch1 = pd.read_parquet('../Data/RegressionPatchVisits/' + title + 'Visit1.parquet', engine='pyarrow')
        Visits_Patch2 = pd.read_parquet('../Data/RegressionPatchVisits/' + title + 'Visit2.parquet', engine='pyarrow')
        dfs.append(Visits_Patch1)
        dfs.append(Visits_Patch2)
        
    VISIT = pd.concat(dfs, ignore_index=False)
    VISIT = VISIT[VISIT['distance'] >= 0.1]
    VISIT['interc'] = 1
    
    return VISIT

def main():
    for session, i in zip(list(short_sessions.itertuples()), range(len(short_sessions))):
        title = 'ShortSession'+str(i)

        mouse_pos = pd.read_parquet('../Data/MousePos/' + title + 'mousepos.parquet', engine='pyarrow')
        states = np.load('../Data/HMMStates/' + title+'States_Unit.npy', allow_pickle=True)
        mouse_pos['states'] = pd.Series(states, index=mouse_pos.index)
        
        Visits_Patch1 = patch.Visits(mouse_pos, patch = 'Patch1', pre_period_seconds = 30)
        Visits_Patch2 = patch.Visits(mouse_pos, patch = 'Patch2', pre_period_seconds = 30)
        Visits_Patch1, Visits_Patch2 = patch.VisitIntervals(Visits_Patch1, Visits_Patch2) 

        Visits_Patch1.to_parquet('../Data/RegressionPatchVisits/' + title+'Visit1.parquet', engine='pyarrow')
        Visits_Patch2.to_parquet('../Data/RegressionPatchVisits/' + title+'Visit2.parquet', engine='pyarrow')
        
        print(title)

    VISIT = ConcatenateSessions()
    VISIT.to_parquet('../Data/RegressionPatchVisits/VISIT.parquet', engine='pyarrow')

if __name__ == "__main__":
    main()