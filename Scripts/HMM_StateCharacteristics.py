import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from collections import Counter

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

import SSM.ssm as ssm
from SSM.ssm.util import find_permutation
from SSM.ssm.plots import gradient_cmap, white_to_color_cmap

root = [Path("/ceph/aeon/aeon/data/raw/AEON2/experiment0.2")]

subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])
short_sessions = sessions.iloc[[4,16,17,20,23,24,25,28,29,30,31]]
long_sessions = sessions.iloc[[8, 10, 11, 14]]

feature = ['smoothed_speed', 'smoothed_acceleration','r']
color_names = ['black', "blue", "red", "tan", "green", "brown", "purple", "orange",'turquoise', "black"]

def PelletDeliveries(t = 6, n=9):
    
    Pellets_State = []
    
    for session, j in zip(list(short_sessions.itertuples()), range(len(short_sessions))):
        title = 'ShortSession'+str(j)
        
        mouse_pos = pd.read_parquet('../Data/MousePos/' + title + 'mousepos.parquet', engine='pyarrow')
        
        states = np.load('../Data/HMMStates/'+ title + 'States_Unit.npy', allow_pickle=True)
        mouse_pos['states'] = pd.Series(states, index = mouse_pos.index)
        
        try:
            PELLET = pd.read_parquet('../Data/Pellets/'+title+'PELLET.parquet', engine='pyarrow')
        except FileNotFoundError:
            start, end = mouse_pos.index[0], mouse_pos.index[-1]
                        
            pellets_patch1 = api.load(root, exp02.Patch1.DeliverPellet, start=start, end=end)
            pellets_patch2 = api.load(root, exp02.Patch2.DeliverPellet, start=start, end=end)
            
            PELLET = pd.concat([pellets_patch1,pellets_patch2], ignore_index=False)
            PELLET.to_parquet('../Data/Pellets/'+title+'PELLET.parquet', engine='pyarrow')


        for i in range(len(PELLET)):
            trigger = PELLET.index[i]
            
            latest_valid_index = mouse_pos.loc[trigger - pd.Timedelta('6S'):trigger, 'states'].index
            latest_valid_state = mouse_pos.loc[latest_valid_index, ['states']].values.reshape(-1)
            if len(latest_valid_state) >= 50: latest_valid_state  = latest_valid_state[-50:]
            
            next_valid_index = mouse_pos.loc[trigger:trigger + pd.Timedelta('6S'), 'states'].index
            next_valid_state = mouse_pos.loc[next_valid_index, ['states']].values.reshape(-1)
            if len(next_valid_state) >= 50: next_valid_state  = next_valid_state[:50]
            
            state = np.concatenate((latest_valid_state, next_valid_state))
            
            Pellets_State.append(state)


    N = n
    colors = sns.xkcd_palette(color_names[0:N])
    cmap = gradient_cmap(colors)

    fig, axs = plt.subplots(1, 1, figsize=(6, 16))
    sns.heatmap(Pellets_State,cmap=cmap, ax=axs, vmin=0, vmax = N-1, cbar = True)
    axs.set_aspect('auto')

    axs.set_xticks([50])
    axs.set_xticklabels(['Pellet'], rotation = 0)

    axs.set_ylabel("Pellet Deliveries")
    axs.set_yticks([])

    plt.savefig('../Images/HMM_States/PelletDelivery.png')
    plt.show()
    
def StateBeforeVisit(n=9):
    STATE, DURATION = [],[]
    for session, j in zip(list(short_sessions.itertuples()), range(len(short_sessions))):
        title = 'ShortSession'+str(j)
        
        mouse_pos = pd.read_parquet('../Data/MousePos/' + title + 'mousepos.parquet', engine='pyarrow')
        states = np.load('../Data/HMMStates/'+ title + 'States_Unit.npy', allow_pickle=True)
        mouse_pos['states'] = pd.Series(states, index = mouse_pos.index)
        
        Visits_Patch1 = pd.read_parquet('../Data/RegressionPatchVisits/' + title + 'Visit1.parquet', engine='pyarrow')
        Visits_Patch2 = pd.read_parquet('../Data/RegressionPatchVisits/' + title + 'Visit2.parquet', engine='pyarrow')
        VISIT = pd.concat([Visits_Patch1,Visits_Patch2], ignore_index=False)
        VISIT = VISIT[VISIT['distance'] >= 0.1]
        
        for i in range(len(VISIT)):
            trigger = VISIT.iloc[i]['start']
            
            latest_valid_index = mouse_pos.loc[trigger - pd.Timedelta('11S'):trigger, 'states'].index
            latest_valid_state = mouse_pos.loc[latest_valid_index, ['states']].values.reshape(-1)
            if len(latest_valid_state) >= 100: latest_valid_state  = latest_valid_state[-100:]
            
            count = Counter(latest_valid_state)
            states_before_forage, frequency = count.most_common(1)[0]
            
            STATE.append(states_before_forage)
            DURATION.append(VISIT.iloc[i]['duration'])
        
    fig, axs = plt.subplots(1, 1, figsize=(10,6))
    axs.scatter(STATE,DURATION)
    axs.set_xlabel('State')
    axs.set_ylabel("Visit Duration")
    axs.set_yticks([])
    plt.savefig('../Images/HMM_States/StateBeforeVisit.png')
    plt.show()


def StartVisit(n=9):    
    STATES, DURATION = [],[]
    for session, j in zip(list(short_sessions.itertuples()), range(len(short_sessions))):
        title = 'ShortSession'+str(j)
        
        mouse_pos = pd.read_parquet('../Data/MousePos/' + title + 'mousepos.parquet', engine='pyarrow')
        states = np.load('../Data/HMMStates/'+ title + 'States_Unit.npy', allow_pickle=True)
        mouse_pos['states'] = pd.Series(states, index = mouse_pos.index)
        
        Visits_Patch1 = pd.read_parquet('../Data/RegressionPatchVisits/' + title + 'Visit1.parquet', engine='pyarrow')
        Visits_Patch2 = pd.read_parquet('../Data/RegressionPatchVisits/' + title + 'Visit2.parquet', engine='pyarrow')
        VISIT = pd.concat([Visits_Patch1,Visits_Patch2], ignore_index=False)
        VISIT = VISIT[VISIT['distance'] >= 0.1]
        
        for i in range(len(VISIT)):
            trigger = VISIT.iloc[i]['start']
            
            latest_valid_index = mouse_pos.loc[trigger - pd.Timedelta('21S'):trigger, 'states'].index
            latest_valid_state = mouse_pos.loc[latest_valid_index, ['states']].values.reshape(-1)
            if len(latest_valid_state) >= 200: latest_valid_state  = latest_valid_state[-200:]
            
            next_valid_index = mouse_pos.loc[trigger:trigger + pd.Timedelta('6S'), 'states'].index
            next_valid_state = mouse_pos.loc[next_valid_index, ['states']].values.reshape(-1)
            if len(next_valid_state) >= 50: next_valid_state  = next_valid_state[:50]
            
            state = np.concatenate((latest_valid_state, next_valid_state))
            
            STATES.append(state)
            DURATION.append(VISIT.iloc[i]['duration'])

    index = np.argsort(DURATION)
    STATES = np.array(STATES)[index]
    
    N = n
    colors = sns.xkcd_palette(color_names[0:N])
    cmap = gradient_cmap(colors)

    fig, axs = plt.subplots(1, 1, figsize=(10, 16))
    sns.heatmap(STATES,cmap=cmap, ax=axs, vmin=0, vmax = N-1, cbar = True)
    axs.set_aspect('auto')
    
    axs.set_xticks([200])
    axs.set_xticklabels(['Enter'], rotation = 0)

    axs.set_ylabel("Visits")
    axs.set_yticks([])

    plt.savefig('../Images/HMM_States/EnterVisit.png')
    plt.show()
    
    
    AVE_STATES = []
    AVE_STATES_D = []
    for k in np.arange(n):
        index = STATES == k
        states = index*1
        AVE_STATES.append(np.mean(states, axis = 0))
        AVE_STATES_D.append(np.mean(states, axis = 1))
    
    fig, axs = plt.subplots(1, 1, figsize=(20, 4))
    sns.heatmap(AVE_STATES,ax=axs)
    axs.set_aspect('auto')
    
    axs.set_xticks([200])
    axs.set_xticklabels(['Enter'], rotation = 0)

    axs.set_ylabel("Visits")
    axs.set_yticks([])

    plt.savefig('../Images/HMM_States/EnterVisit' + 'EachState' + '.png')
    plt.show()
    
    
    fig, axs = plt.subplots(1, 1, figsize=(4, 16))
    sns.heatmap(np.array(AVE_STATES_D).T,ax=axs)
    axs.set_aspect('auto')

    axs.set_ylabel("Visits")
    axs.set_yticks([])

    plt.savefig('../Images/HMM_States/EnterVisit' + 'EachState_D' + '.png')
    plt.show()
    
def EndVisit(n=9):    
    STATES, DURATION = [],[]
    for session, j in zip(list(short_sessions.itertuples()), range(len(short_sessions))):
        title = 'ShortSession'+str(j)
        
        mouse_pos = pd.read_parquet('../Data/MousePos/' + title + 'mousepos.parquet', engine='pyarrow')
        states = np.load('../Data/HMMStates/'+ title + 'States_Unit.npy', allow_pickle=True)
        mouse_pos['states'] = pd.Series(states, index = mouse_pos.index)
        
        Visits_Patch1 = pd.read_parquet('../Data/RegressionPatchVisits/' + title + 'Visit1.parquet', engine='pyarrow')
        Visits_Patch2 = pd.read_parquet('../Data/RegressionPatchVisits/' + title + 'Visit2.parquet', engine='pyarrow')
        VISIT = pd.concat([Visits_Patch1,Visits_Patch2], ignore_index=False)
        VISIT = VISIT[VISIT['distance'] >= 0.1]
        
        for i in range(len(VISIT)):
            trigger = VISIT.iloc[i]['end']
            
            latest_valid_index = mouse_pos.loc[trigger - pd.Timedelta('6S'):trigger, 'states'].index
            latest_valid_state = mouse_pos.loc[latest_valid_index, ['states']].values.reshape(-1)
            if len(latest_valid_state) >= 50: latest_valid_state  = latest_valid_state[-50:]
            
            next_valid_index = mouse_pos.loc[trigger:trigger + pd.Timedelta('21S'), 'states'].index
            next_valid_state = mouse_pos.loc[next_valid_index, ['states']].values.reshape(-1)
            if len(next_valid_state) >= 200: next_valid_state  = next_valid_state[:200]
            state = np.concatenate((latest_valid_state, next_valid_state))
            STATES.append(state)
            DURATION.append(VISIT.iloc[i]['duration'])

    index = np.argsort(DURATION)
    STATES = np.array(STATES)[index]
    N = n
    colors = sns.xkcd_palette(color_names[0:N])
    cmap = gradient_cmap(colors)


    for k in [1,3,5,6,7,8]:
        index = STATES == k
        states = index*k
        fig, axs = plt.subplots(1, 1, figsize=(10, 16))
        sns.heatmap(states,cmap=cmap, ax=axs, vmin=0, vmax = N-1, cbar = True)
        axs.set_aspect('auto')
        
        axs.set_xticks([50])
        axs.set_xticklabels(['Leave'], rotation = 0)

        axs.set_ylabel("Visits")
        axs.set_yticks([])

        plt.savefig('../Images/HMM_States/EndVisit' + str(k) + '.png')
        plt.show()
    
    fig, axs = plt.subplots(1, 1, figsize=(10, 16))
    sns.heatmap(STATES,cmap=cmap, ax=axs, vmin=0, vmax = N-1, cbar = True)
    axs.set_aspect('auto')
    
    axs.set_xticks([50])
    axs.set_xticklabels(['Leave'], rotation = 0)

    axs.set_ylabel("Visits")
    axs.set_yticks([])

    plt.savefig('../Images/HMM_States/EndVisit.png')
    plt.show()
        

def main():
    
    #PelletDeliveries()
    #StateBeforeVisit()
    StartVisit()
    #EndVisit()
    
    
    
if __name__ == "__main__":
    main()