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

import Functions.HMM as HMM
import Functions.kinematics as kinematics
import Functions.patch as patch

root = [Path("/ceph/aeon/aeon/data/raw/AEON2/experiment0.2")]

subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])
short_sessions = sessions.iloc[[4,16,17,20,23,24,25,26,28,29,30,31]]
long_sessions = sessions.iloc[[8, 10, 11, 14]]

feature = ['weight', 'smoothed_speed', 'smoothed_acceleration']

def FindModelsShort():

    for session, j in zip(list(short_sessions.itertuples()), range(len(short_sessions))):
        title = 'ShortSession'+str(j)
        
        mouse_pos = pd.read_parquet('../Data/MousePos' + title + 'mousepos.parquet', engine='pyarrow')
    
        dfs = []
        for k in range(len(short_sessions)):
            if k == j: continue
            title = 'ShortSession'+str(k)
            mouse_pos = pd.read_parquet('../Data/MousePos' + title + 'mousepos.parquet', engine='pyarrow')
            dfs.append(mouse_pos)
        MOUSE_POS = dfs[0]
        for df in dfs[1:]: MOUSE_POS = MOUSE_POS.add(df, fill_value=0)
        
        obs = np.array(mouse_pos[feature])
        OBS = np.array(MOUSE_POS[feature])
        
        LL = []
        N = np.arange(3,11,1)
        for n in N:
            hmm, states, transition_mat, lls = HMM.FitHMM(obs, num_states = n, n_iters = 50)
            ll = hmm.log_likelihood(OBS)
            LL.append(ll)

        fig, axs = plt.subplots(1,1,figsize = (10,7))
        axs.scatter(N, LL)
        axs.plot(N, LL, color = 'black')
        axs.set_xticks(N)
        plt.savefig('../Images/HMM_StateChoice/' + title+'.png')
        plt.show()


def main():
    # For short sessions:
    FindModelsShort()
    
    # For long sessions
    #FitModelsLong()
    
if __name__ == "__main__":
    main()