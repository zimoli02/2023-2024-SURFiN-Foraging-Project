import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

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

color_names = ['black', "blue", "red", "tan", "green", "brown", "purple", "orange", 'turquoise', "yellow", 'pink', 'darkblue']

parent_dir = current_script_path.parents[2] / 'aeon_mecha' 
sys.path.insert(0, str(parent_dir))
import aeon
import aeon.io.api as api
from aeon.schema.schemas import social02
from aeon.analysis.utils import visits, distancetravelled


def main():
    '''type, mouse = 'Post','BAA-1104045'
    PELLET = pd.read_parquet('../SocialData/Pellets/'+ 'Post' + "_" + 'BAA-1104045' +'_PELLET.parquet', engine='pyarrow')

    Pellets_State = []
    n=12
    mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')

    features = ['smoothed_speed', 'smoothed_acceleration', 'r','bodylength', 'bodyangle', 'nose']
    obs = np.array(mouse_pos[features])
    hmm, states, transition_mat, lls = HMM.FitHMM(obs[0:10*60*60], num_states = n, n_iters = 50)
    states = hmm.most_likely_states(obs)
    
    state_mean_speed = hmm.observations.params[0].T[0]
    index = np.argsort(state_mean_speed, -1)     
        
    new_values = np.empty_like(states)
    for i, val in enumerate(index): new_values[states == val] = i
    states = new_values
    
    mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)

    for i in range(len(PELLET)):
        trigger = PELLET.index[i]
        
        latest_valid_index = mouse_pos.loc[trigger - pd.Timedelta('6S'):trigger, 'state'].index
        latest_valid_state = mouse_pos.loc[latest_valid_index, ['state']].values.reshape(-1)
        if len(latest_valid_state) >= 50: latest_valid_state  = latest_valid_state[-50:]
        
        next_valid_index = mouse_pos.loc[trigger:trigger + pd.Timedelta('6S'), 'state'].index
        next_valid_state = mouse_pos.loc[next_valid_index, ['state']].values.reshape(-1)
        if len(next_valid_state) >= 50: next_valid_state  = next_valid_state[:50]
        
        state = np.concatenate((latest_valid_state, np.array([np.nan]), next_valid_state))
        
        if len(state) == 101: Pellets_State.append(state)


    N = 12
    colors = sns.xkcd_palette(color_names[0:N])
    cmap = gradient_cmap(colors)

    fig, axs = plt.subplots(1, 1, figsize=(6, 16))
    sns.heatmap(Pellets_State,cmap=cmap, ax=axs, vmin=0, vmax = N-1, cbar = True)
    axs.set_aspect('auto')

    axs.set_xticks([50])
    axs.set_xticklabels(['Pellet'], rotation = 0)

    axs.set_ylabel("Pellet Deliveries")
    axs.set_yticks([])

    plt.savefig('PelletDelivery.png')
    plt.show()'''
    
    root = '/ceph/aeon/aeon/data/raw/AEON3/social0.2'
    
    
    start, end = pd.Timestamp('2024-02-05 15:43:07.00'), pd.Timestamp('2024-02-08 15:00:00.00')
    encoder = aeon.load(root, social02.Patch1.Encoder, start=start, end=end)
    encoder = encoder[::5]
    
 
    encoder = encoder.sort_index()
    encoder = encoder[~encoder.index.duplicated(keep='first')]

    
    w = -distancetravelled(encoder.angle).to_numpy()
    dw = np.concatenate((np.array([0]), w[:-1]- w[1:]))
    encoder['Distance'] = pd.Series(w, index=encoder.index)
    encoder['DistanceChange'] = pd.Series(dw, index=encoder.index)
    encoder['DistanceChange'] = encoder['DistanceChange'].rolling('10S').mean()
    encoder['Move'] = np.where(abs(encoder.DistanceChange) > 0.001, 1, 0)
    
    groups = encoder['Move'].ne(encoder['Move'].shift()).cumsum()
    one_groups = encoder[encoder['Move'] == 1].groupby(groups).groups
    one_groups = list(one_groups.values())
    
    D = []

    for i in range(len(one_groups) - 1):
        end_current_group = one_groups[i][-1]
        start_next_group = one_groups[i + 1][0]
        duration = start_next_group - end_current_group
        D.append(duration.total_seconds())
    D = np.array(D)
    plt.hist(D[D<20], bins = 100)
    plt.savefig('Duration.png')
    plt.show()
    
if __name__ == "__main__":
    main()