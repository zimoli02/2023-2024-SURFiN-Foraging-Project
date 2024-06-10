import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from pathlib import Path

import sys
from pathlib import Path

current_script_path = Path(__file__).resolve()
function_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(function_dir))
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

root = '/ceph/aeon/aeon/data/raw/AEON3/social0.2/'

nodes_name = ['nose', 'head', 'right_ear', 'left_ear', 'spine1', 'spine2','spine3', 'spine4']

INFO = pd.read_parquet('../SocialData/INFO3.parquet', engine='pyarrow')
TYPE = ['Pre','Post']
MOUSE = ['BAA-1104045', 'BAA-1104047']
LABELS = [
    ['Pre','BAA-1104045'],
    ['Pre','BAA-1104047'],
    ['Post','BAA-1104045'],
    ['Post','BAA-1104047']
]

color_names = ['black', "blue", "red", "tan", "green", "brown", "purple", "orange", 'turquoise', "yellow", 'pink', 'darkblue']

def FixNan(mouse_pos, column):
    mouse_pos[column] = mouse_pos[column].interpolate()
    mouse_pos[column] = mouse_pos[column].ffill()
    mouse_pos[column] = mouse_pos[column].bfill()
    
    return mouse_pos

def Calculate_TransM_From_States(N):
    for label in LABELS:
        type, mouse = label[0], label[1]
        mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
        states = np.load('../SocialData/HMMStates_Body/' + type + "_" + mouse + "_States.npy", allow_pickle = True)
        mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)
        
        grouped = mouse_pos.groupby([pd.Grouper(freq='10S'), 'state']).size()
        prob = grouped.groupby(level=0).apply(lambda g: g / g.sum())
        states_prob = prob.unstack(level=-1).fillna(0)
        states_prob.index = states_prob.index.get_level_values(0)
        
        manual_trans_mat = np.zeros((N,N))
        for k in range(1,len(states)): manual_trans_mat[states[k-1]][states[k]] += 1
        for k in range(N): 
            if np.sum(manual_trans_mat[k]) != 0: manual_trans_mat[k] = manual_trans_mat[k]/np.sum(manual_trans_mat[k])

        HMM.PlotTransition(manual_trans_mat, title = '../Images/Social_HMM_Body/'+ type + "_" + mouse +'TransM.png')
        

def Get_Latent_States(id, n, features):
    type, mouse = LABELS[id][0], LABELS[id][1]
    mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
    mouse_pos = mouse_pos[pd.Timestamp('2024-02-06 07:00:00.0'):pd.Timestamp('2024-02-07 07:00:00.0')]
    
    obs = np.array(mouse_pos[features])
    hmm, states, transition_mat, lls = HMM.FitHMM(obs, num_states = n, n_iters = 50)
    
    state_mean_speed = hmm.observations.params[0].T[0]
    index = np.argsort(state_mean_speed, -1)     
    
    HMM.PlotTransition(transition_mat[index].T[index].T, title = '../Images/Social_HMM_Body/TransM.png')
    np.save('../SocialData/HMMStates_Body/TransM.npy', transition_mat[index].T[index].T)
    
    for label in LABELS:
        type, mouse = label[0], label[1]
        mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
        obs = np.array(mouse_pos[features])
        
        states = hmm.most_likely_states(obs)
        
        new_values = np.empty_like(states)
        for i, val in enumerate(index): new_values[states == val] = i
        states = new_values
        np.save('../SocialData/HMMStates_Body/' + type + "_" + mouse + "_States.npy", states)

def PlotStatesWithTime(states, mouse_pos, type, N, axs):
    colors = sns.xkcd_palette(color_names[0:N])
    cmap = gradient_cmap(colors)

    times = pd.to_datetime(mouse_pos.index)
    numerical_times = (times - times[0]).total_seconds().values
    states_array = states.reshape(1, -1)
    
    
    extent = [numerical_times[0], numerical_times[-1], 0, 1]
    cax = axs.imshow(states_array, aspect="auto", cmap=cmap, vmin=0, vmax=N-1, extent=extent)
    
    axs.set_xlabel('Time')
    axs.set_xticks(numerical_times[::len(numerical_times)//10])
    axs.set_xticklabels([time.strftime('%H:%M:%S') for time in times[::len(times)//10]], rotation=45, ha='right')
    
    axs.set_ylabel(type)
    axs.set_yticks([])

    return cax, axs

def PlotStates(mouse_pos, N, type, mouse):
    grouped = mouse_pos.groupby([pd.Grouper(freq='10S'), 'state']).size()
    prob = grouped.groupby(level=0).apply(lambda g: g / g.sum())
    states_prob = prob.unstack(level=-1).fillna(0)
    states_prob.index = states_prob.index.get_level_values(0)
        
    for i in range(N):
        if i not in states_prob.columns: states_prob[i] = 0
        
    x = np.array([np.array(states_prob[i].to_list()) for i in range(N)])

    states_ = []
    for i in range(len(x[0])):
        current_state = x.T[i]
        states_.append(np.argmax(current_state))
        
    fig, axs = plt.subplots(1, 1, figsize=(35, 4))
    cax, axs = PlotStatesWithTime(np.array(states_), states_prob, type = 'States Prob.', N = N, axs = axs)
    cbar = fig.colorbar(cax, ax=axs, orientation='vertical')
    cbar.set_ticks(np.arange(0, N))
    cbar.set_ticklabels([f'State {val}' for val in np.arange(0, N)])
    plt.savefig('../Images/Social_HMM_Body/' + type + "_" +mouse+'.png')
    plt.show()


def Display_Latent_States(N):
    X, Y, SPEED, ACCE, LENGTH, ANGLE, NOSE = [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)]
    for label in LABELS:
        type, mouse = label[0], label[1]
        mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
        states = np.load('../SocialData/HMMStates_Body/' + type + "_" + mouse + "_States.npy", allow_pickle = True)
        mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)
        
        PlotStates(mouse_pos, N, type, mouse)

        for i in range(N):
            X[i] = np.concatenate([X[i], mouse_pos['smoothed_position_x'][states==i]])
            Y[i] = np.concatenate([Y[i], mouse_pos['smoothed_position_y'][states==i]])
            SPEED[i] = np.concatenate([SPEED[i], mouse_pos['smoothed_speed'][states==i]])
            ACCE[i] = np.concatenate([ACCE[i], mouse_pos['smoothed_acceleration'][states == i]])
            LENGTH[i] = np.concatenate([LENGTH[i], mouse_pos['bodylength'][states == i]])
            ANGLE[i] = np.concatenate([ANGLE[i], mouse_pos['bodyangle'][states == i]])
            NOSE[i] = np.concatenate([NOSE[i], mouse_pos['nose'][states == i]])
        
    fig, axs = plt.subplots(1, N, figsize = (N*8-2,6))
    for i in range(N):
        axs[i].scatter(X[i], Y[i], color = color_names[i], s = 2, alpha = 0.2)
        axs[i].set_xlim((100,1400))
        axs[i].set_ylim((-20,1100))
        axs[i].set_title('State' + str(i))
        axs[i].set_xlabel('X')
        axs[i].set_ylabel('Y')
    plt.savefig('../Images/Social_HMM_Body/Position.png')
    plt.show()
    
    DATA = [SPEED, ACCE, LENGTH, ANGLE, NOSE]
    FEATURE = ['SPEED', 'ACCE', 'LENGTH', 'ANGLE', 'NOSE']
    fig, axs = plt.subplots(len(FEATURE), 1, figsize = (10, len(FEATURE)*7-1))
    for data, i in zip(DATA, range(len(DATA))):
        means = [np.mean(arr) for arr in data]
        var = [np.std(arr)/np.sqrt(len(arr)) for arr in data]
        axs[i].bar(range(N), means, yerr=var, capsize=5)
        axs[i].set_xticks(range(0, N), [str(j) for j in range(N)])
        axs[i].set_ylabel(FEATURE[i])
    plt.savefig('../Images/Social_HMM_Body/Data.png')
    plt.show()


def Get_Latent_State_Number(features, N):
    LogLikelihood = []
    
    for i in range(len(LABELS)):
        type, mouse = LABELS[i][0], LABELS[i][1]
        mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
        start = mouse_pos.index[0]
        mouse_pos = mouse_pos[start:start + pd.Timedelta('1H')]
        
        MOUSE_POS = []
        for j in range(len(LABELS)):
            if i == j: continue
            else:
                type_, mouse_ = LABELS[j][0], LABELS[j][1]
                mouse_pos_ = pd.read_parquet('../SocialData/HMMData/' + type_ + "_" + mouse_ + '.parquet', engine='pyarrow')
                MOUSE_POS.append(mouse_pos_)
        MOUSE_POS = pd.concat(MOUSE_POS, ignore_index=False)
        
        obs = np.array(mouse_pos[features])
        OBS = np.array(MOUSE_POS[features])

        loglikelihood = []
        for n in N:
            hmm, states, transition_mat, lls = HMM.FitHMM(obs, num_states = n, n_iters = 50)
            ll = hmm.log_likelihood(OBS)
            loglikelihood.append(ll/len(OBS[0]))
        
        LogLikelihood.append(loglikelihood)
        np.save('../SocialData/HMMData/LogLikelihood_Body.npy', LogLikelihood)

def Display_Latent_State_Number(N):    
    LogLikelihood = np.load('../SocialData/HMMData/LogLikelihood_Body.npy', allow_pickle=True)
    fig, axs = plt.subplots(1,1,figsize = (10,7))
    for i in range(len(LogLikelihood)):
        loglikelihood = LogLikelihood[i]
        axs.scatter(N, loglikelihood)
        axs.plot(N, loglikelihood, label = i)
    axs.set_xticks(N)
    axs.legend()
    plt.savefig('../Images/Social_HMM/StateNumber_Body.png')
    plt.show()

def Get_Observations(kinematics_Update, weight_Update, patch_Update, r_Update, bodylength_Update, bodyangle_Update, nose_Update):
    for type in TYPE:
        for mouse in MOUSE:
            try:
                mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
            except FileNotFoundError:
                mouse_pos = Get_Data(type, mouse)
            
            data_x = pd.read_parquet('../SocialData/BodyData/' + type + "_" + mouse + '_x.parquet', engine='pyarrow')
            data_y = pd.read_parquet('../SocialData/BodyData/' + type + "_" + mouse + '_y.parquet', engine='pyarrow')

            if patch_Update:
                mouse_pos = patch.InPatch(mouse_pos, r = 40, interval = 5, patch_loc = [[910.25, 544],[613.75, 724],[604.5, 375.75]])
            if r_Update:
                mouse_pos = patch.Radius(mouse_pos, x_o = 709.4869937896729, y_o = 546.518087387085)
            if bodylength_Update:
                dx = data_x['spine4'] - data_x['head']
                dy = data_y['spine4'] - data_y['head']
                d = np.sqrt(dx**2 + dy**2)
                mouse_pos['bodylength'] = pd.Series(d, index=mouse_pos.index)
                mouse_pos = FixNan(mouse_pos,'bodylength')
            if bodyangle_Update:
                head = np.array([data_x['head'], data_y['head']]).T
                spine2 = np.array([data_x['spine2'], data_y['spine2']]).T
                spine4 = np.array([data_x['spine4'], data_y['spine4']]).T
                v1 = head - spine2
                v2 = spine4 - spine2
                dot_product = np.einsum('ij,ij->i', v1, v2)
                norm1 = np.linalg.norm(v1, axis=1)
                norm2 = np.linalg.norm(v2, axis=1)
                cos_theta = dot_product / (norm1 * norm2)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                radians_theta = np.arccos(cos_theta)
                degree_theta = np.degrees(radians_theta)
                mouse_pos['bodyangle'] = pd.Series(degree_theta, index=mouse_pos.index)
                mouse_pos = FixNan(mouse_pos,'bodyangle')
            if nose_Update:
                mid_x = (data_x['right_ear'] + data_x['left_ear'])/2
                mid_y = (data_y['right_ear'] + data_y['left_ear'])/2
                dx = data_x['nose'] - mid_x
                dy = data_y['nose'] - mid_y
                d = np.sqrt(dx**2 + dy**2)
                mouse_pos['nose'] = pd.Series(d, index=mouse_pos.index)
                mouse_pos = FixNan(mouse_pos,'nose')
                
            mouse_pos.to_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')

def Get_Data(Type, Mouse):
    Mouse_pos = []
    for i in range(len(INFO)):
        type, mouse, start = INFO.loc[i, 'Type'], INFO.loc[i, 'Mouse'], INFO.loc[i, 'Start']
        if mouse == Mouse and type == Type: 
            mouse_pos = pd.read_parquet('../SocialData/LDSData/' + start + '.parquet', engine='pyarrow')
            smoothRes = np.load('../SocialData/LDS/' + start +'_smoothRes.npz')
            kinematics.AddKinematics(smoothRes, mouse_pos)
            Mouse_pos.append(mouse_pos)
    
    Mouse_pos = pd.concat(Mouse_pos, ignore_index=False)
    Mouse_pos.to_parquet('../SocialData/HMMData/' + Type + "_" + Mouse + '.parquet', engine='pyarrow')

    return Mouse_pos

def PelletDeliveries(t = 6, n=9):
    
    Pellets_State = []
    
    for label in LABELS:
        type, mouse = label[0], label[1]
        mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
        states = np.load('../SocialData/HMMStates_Body/' + type + "_" + mouse + "_States.npy", allow_pickle = True)
        mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)
        
        try:
            PELLET = pd.read_parquet('../SocialData/Pellets/'+ type + "_" + mouse +'_PELLET.parquet', engine='pyarrow')
        except FileNotFoundError:
            start, end = mouse_pos.index[0], mouse_pos.index[-1]
            
            social02.Patch1.DeliverPellet.value = 1
            pellets_patch1 = aeon.load(root, social02.Patch1.DeliverPellet, start=start, end=end)
            
            social02.Patch2.DeliverPellet.value = 1
            pellets_patch2 = aeon.load(root, social02.Patch2.DeliverPellet, start=start, end=end)
            
            social02.Patch3.DeliverPellet.value = 1
            pellets_patch3 = aeon.load(root, social02.Patch3.DeliverPellet, start=start, end=end)
            
            PELLET = pd.concat([pellets_patch1,pellets_patch2, pellets_patch3], ignore_index=False)
            PELLET = PELLET.sort_index()
            PELLET.to_parquet('../SocialData/Pellets/'+ type + "_" + mouse +'_PELLET.parquet', engine='pyarrow')


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

    plt.savefig('../Images/Social_HMM_Body/PelletDelivery.png')
    plt.show()
    
def StateBeforeVisit(n=9):
    STATE, DURATION = [],[]
    for label in LABELS:
        type, mouse = label[0], label[1]
        mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
        states = np.load('../SocialData/HMMStates_Body/' + type + "_" + mouse + "_States.npy", allow_pickle = True)
        mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)
        
        Visits = pd.read_parquet('../SocialData/VisitData/'  + type + "_" + mouse +'_Visit.parquet', engine='pyarrow')
        Visits = Visits.dropna(subset=['speed'])
        Visits['distance'] = abs(Visits['distance'])
        Visits = Visits[Visits['distance'] >= 0.1]
        
        for i in range(len(Visits)):
            trigger = Visits.iloc[i]['start']
            
            latest_valid_index = mouse_pos.loc[trigger - pd.Timedelta('11S'):trigger, 'state'].index
            latest_valid_state = mouse_pos.loc[latest_valid_index, ['state']].values.reshape(-1)
            if len(latest_valid_state) >= 100: latest_valid_state  = latest_valid_state[-100:]
            
            count = Counter(latest_valid_state)
            states_before_forage, frequency = count.most_common(1)[0]
            
            #if len(states_before_forage) == 0: continue
            STATE.append(states_before_forage)
            DURATION.append(Visits.iloc[i]['duration'])
        
    fig, axs = plt.subplots(1, 1, figsize=(10,6))
    axs.scatter(STATE,DURATION)
    axs.set_xlabel('State')
    axs.set_ylabel("Visit Duration")
    axs.set_yticks([])
    plt.savefig('../Images/Social_HMM_Body/StateBeforeVisit.png')
    plt.show()


def StartVisit(n=9):    
    STATE, DURATION = [],[]
    for label in LABELS:
        type, mouse = label[0], label[1]
        mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
        states = np.load('../SocialData/HMMStates_Body/' + type + "_" + mouse + "_States.npy", allow_pickle = True)
        mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)
        
        Visits = pd.read_parquet('../SocialData/VisitData/'  + type + "_" + mouse +'_Visit.parquet', engine='pyarrow')
        Visits = Visits.dropna(subset=['speed'])
        Visits['distance'] = abs(Visits['distance'])
        Visits = Visits[Visits['distance'] >= 0.1]
        
        for i in range(len(Visits)):
            trigger = Visits.iloc[i]['start']
            
            latest_valid_index = mouse_pos.loc[trigger - pd.Timedelta('21S'):trigger, 'state'].index
            latest_valid_state = mouse_pos.loc[latest_valid_index, ['state']].values.reshape(-1)
            if len(latest_valid_state) >= 200: latest_valid_state  = latest_valid_state[-200:]
            
            next_valid_index = mouse_pos.loc[trigger:trigger + pd.Timedelta('6S'), 'state'].index
            next_valid_state = mouse_pos.loc[next_valid_index, ['state']].values.reshape(-1)
            if len(next_valid_state) >= 50: next_valid_state  = next_valid_state[:50]
            state = np.concatenate((latest_valid_state, np.array([np.nan]), next_valid_state))
            
            if len(state) == 251: 
                STATE.append(state)
                DURATION.append(Visits.iloc[i]['duration'])

    index = np.argsort(DURATION)
    STATE = np.array(STATE)[index]
    
    N = n
    colors = sns.xkcd_palette(color_names[0:N])
    cmap = gradient_cmap(colors)

    fig, axs = plt.subplots(1, 1, figsize=(10, 16))
    sns.heatmap(STATE, cmap=cmap, ax=axs, vmin=0, vmax = N-1, cbar = True)
    axs.set_aspect('auto')
    
    axs.set_xticks([200])
    axs.set_xticklabels(['Enter'], rotation = 0)

    axs.set_ylabel("Visits")
    axs.set_yticks([])

    plt.savefig('../Images/Social_HMM_Body/EnterVisit.png')
    plt.show()
    

    AVE_STATES = []
    AVE_STATES_D = []
    for k in np.arange(n):
        index = STATE == k
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

    plt.savefig('../Images/Social_HMM_Body/EnterVisit_' + 'EachState' + '.png')
    plt.show()
    
    '''
    fig, axs = plt.subplots(1, 1, figsize=(4, 16))
    sns.heatmap(np.array(AVE_STATES_D).T,ax=axs)
    axs.set_aspect('auto')

    axs.set_ylabel("Visits")
    axs.set_yticks([])

    plt.savefig('../Images/Social_HMM/EnterVisit_' + 'EachState_D' + '.png')
    plt.show()'''
    
def EndVisit(n=9):    
    STATES, DURATION = [],[]
    for label in LABELS:
        type, mouse = label[0], label[1]
        mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
        states = np.load('../SocialData/HMMStates_Body/' + type + "_" + mouse + "_States.npy", allow_pickle = True)
        mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)
        
        Visits = pd.read_parquet('../SocialData/VisitData/'  + type + "_" + mouse +'_Visit.parquet', engine='pyarrow')
        Visits = Visits.dropna(subset=['speed'])
        Visits['distance'] = abs(Visits['distance'])
        Visits = Visits[Visits['distance'] >= 0.1]
        
        '''        
        for i in range(len(Visits)):
            trigger = Visits.iloc[i]['end']
            
            latest_valid_index = mouse_pos.loc[trigger - pd.Timedelta('6S'):trigger, 'state'].index
            latest_valid_state = mouse_pos.loc[latest_valid_index, ['state']].values.reshape(-1)
            if len(latest_valid_state) >= 50: latest_valid_state  = latest_valid_state[-50:]
            
            next_valid_index = mouse_pos.loc[trigger:trigger + pd.Timedelta('21S'), 'state'].index
            next_valid_state = mouse_pos.loc[next_valid_index, ['state']].values.reshape(-1)
            if len(next_valid_state) >= 200: next_valid_state  = next_valid_state[:200]
            state = np.concatenate((latest_valid_state, np.array([0]), next_valid_state))
            if len(state) == 0: continue
            STATES.append(state)
            DURATION.append(Visits.iloc[i]['duration'])
            '''
        
        for i in range(len(Visits)):
            trigger = Visits.iloc[i]['end']
            
            latest_valid_index = mouse_pos.loc[trigger - pd.Timedelta('21S'):trigger, 'state'].index
            latest_valid_state = mouse_pos.loc[latest_valid_index, ['state']].values.reshape(-1)
            if len(latest_valid_state) >= 200: latest_valid_state  = latest_valid_state[-200:]
            
            next_valid_index = mouse_pos.loc[trigger:trigger + pd.Timedelta('6S'), 'state'].index
            next_valid_state = mouse_pos.loc[next_valid_index, ['state']].values.reshape(-1)
            if len(next_valid_state) >= 50: next_valid_state  = next_valid_state[:50]
            
            state = np.concatenate((latest_valid_state, np.array([np.nan]), next_valid_state))
            
            if len(state) == 251: 
                STATES.append(state)
                DURATION.append(Visits.iloc[i]['duration'])

    index = np.argsort(DURATION)
    STATES = np.array(STATES)[index]
    N = n
    colors = sns.xkcd_palette(color_names[0:N])
    cmap = gradient_cmap(colors)
    
    fig, axs = plt.subplots(1, 1, figsize=(10, 16))
    sns.heatmap(STATES,cmap=cmap, ax=axs, vmin=0, vmax = N-1, cbar = True)
    axs.set_aspect('auto')
    
    axs.set_xticks([200])
    axs.set_xticklabels(['Leave'], rotation = 0)

    axs.set_ylabel("Visits")
    axs.set_yticks([])

    plt.savefig('../Images/Social_HMM_Body/EndVisit.png')
    plt.show()


def Get_States_Characterized(pellet_delivery = False,state_before_visit = True,start_visit = True,end_visit = True,N=9):
    if pellet_delivery: PelletDeliveries(n=N)
    if state_before_visit: StateBeforeVisit(n=N)
    if start_visit: StartVisit(n=N)
    if end_visit: EndVisit(n=N)
    

def main():
    
    features = ['bodylength', 'bodyangle', 'nose']
    
    '''Get_Observations(kinematics_Update = False,   
                        weight_Update = False,
                        patch_Update = False,
                        r_Update = True,
                        bodylength_Update = True,
                        bodyangle_Update = True,
                        nose_Update = True)
        

    
    Get_Latent_State_Number(features, N = np.arange(3,16))
    
    Display_Latent_State_Number(N = np.arange(3,16))
    '''
    
    
    Get_Latent_States(id=1, n=12, features = features)
    print('Get_Latent_States Completed')
    
    Display_Latent_States(N = 12)
    print('Display_Latent_States Completed')
    
    Calculate_TransM_From_States(N=12)
    print('Calculate_TransM_From_States Completed')
    
    
    Get_States_Characterized(pellet_delivery = True,
                                state_before_visit = False,
                                start_visit = True,
                                end_visit = True,
                                N=12)
    print('Get_States_Characterized Completed')

if __name__ == "__main__":
        main()