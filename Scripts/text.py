import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

import sys
from pathlib import Path

current_script_path = Path(__file__).resolve()
function_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(function_dir))
import Functions.patch as patch
import Functions.HMM as HMM
import Functions.kinematics as kinematics
import Functions.patch as patch
from SSM.ssm.plots import gradient_cmap

parent_dir = current_script_path.parents[2] / 'aeon_mecha' 
sys.path.insert(0, str(parent_dir))
import aeon
import aeon.io.api as api
from aeon.schema.dataset import exp02
from aeon.analysis.utils import visits
from aeon.schema.schemas import social02
from aeon.analysis.utils import visits, distancetravelled

root = [Path("/ceph/aeon/aeon/data/raw/AEON2/experiment0.2")]

subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])
short_sessions = sessions.iloc[[4,16,17,20,23,24,25,28,29,30,31]]
long_sessions = sessions.iloc[[8, 10, 11, 14]]

feature = ['smoothed_speed', 'smoothed_acceleration', 'r']

def FindModelsShort():
    title = 'ShortSession8'
    mouse_pos = pd.read_parquet('../Data/MousePos/' + title + 'mousepos.parquet', engine='pyarrow')

    dfs = []
    for k in range(len(short_sessions)):
        if k == 8: continue
        title = 'ShortSession'+str(k)
        mouse_pos = pd.read_parquet('../Data/MousePos/' + title + 'mousepos.parquet', engine='pyarrow')
        dfs.append(mouse_pos)
    
    MOUSE_POS = dfs[0]
    for df in dfs[1:]: MOUSE_POS = MOUSE_POS.add(df, fill_value=0)
    
    obs = np.array(mouse_pos[feature])
    OBS = np.array(MOUSE_POS[feature])
    
    LogLikelihood = []
    N = np.arange(3,28,1)
    for n in N:
        hmm, states, transition_mat, lls = HMM.FitHMM(obs, num_states = n, n_iters = 50)
        ll = hmm.log_likelihood(OBS)
        LogLikelihood.append(ll/len(OBS[0]))
    
    np.save('../Data/HMMStates/LogLikelihood_Three.npy', LogLikelihood)


def DisplayModelsShort():
    N = np.arange(3,28,1)
    LogLikelihood = np.load('../Data/HMMStates/LogLikelihood_Three.npy', allow_pickle=True)
    

    fig, axs = plt.subplots(1,1,figsize = (10,7))
    axs.scatter(N, LogLikelihood)
    axs.plot(N, LogLikelihood, color = 'black')
    axs.set_xticks(N)
    plt.savefig('../Images/HMM_StateChoice/LogLikelihood_Three.png')
    plt.show()


def main():

    FindModelsShort()
    DisplayModelsShort()
    
    
if __name__ == "__main__":
    main()