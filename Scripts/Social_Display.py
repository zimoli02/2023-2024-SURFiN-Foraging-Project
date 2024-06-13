import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from collections import Counter
from pathlib import Path

import sys
from pathlib import Path

current_script_path = Path(__file__).resolve()
function_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(function_dir))
import Functions.mouse as mouse
import Functions.HMM as HMM
import Functions.kinematics as kinematics
import Functions.patch as patch
from SSM.ssm.plots import gradient_cmap

parent_dir = current_script_path.parents[2] / 'aeon_mecha' 
sys.path.insert(0, str(parent_dir))
import aeon
import aeon.io.api as api
from aeon.schema.schemas import social02

LABELS = [
    ['Pre','BAA-1104045'],
    ['Pre','BAA-1104047'],
    ['Post','BAA-1104045'],
    ['Post','BAA-1104047']
]
nodes_name = ['nose', 'head', 'right_ear', 'left_ear', 'spine1', 'spine2','spine3', 'spine4']
color_names = ['black', "blue", "red", "tan", "green", "brown", "purple", "orange", 'turquoise', "yellow", 'pink', 'darkblue']



def Display_LDS_Trace(mouse_pos, title):
    fig, axs = plt.subplots(4,1, figsize = (40,24))
    mouse_pos.x.plot(ax = axs[0])
    mouse_pos.y.plot(ax = axs[0])
    mouse_pos.smoothed_position_x.plot(ax = axs[1])
    mouse_pos.smoothed_position_y.plot(ax = axs[1])
    mouse_pos.smoothed_speed.plot(ax = axs[2])
    mouse_pos.smoothed_acceleration.plot(ax = axs[3])
        
    axs[0].set_ylabel('Raw Pos.',fontsize = 16)
    axs[1].set_ylabel("Smoothed Pos.",fontsize = 16)
    axs[2].set_ylabel("Smoothed Vel.",fontsize = 16)
    axs[3].set_ylabel("Smoothed Acc.",fontsize = 16)
    plt.savefig(title)
    
def Display_Kinematics_Distribution_Along_Time(mouse_pos, title):
    start = mouse_pos.index[0]
    end = mouse_pos.index[-1]
    starts, ends = [],[]
    while start < end:
        end_ = start + pd.Timedelta('4H')

        starts.append(start)
        ends.append(end_)
        start = end_ + pd.Timedelta('1S')

    n = len(starts)
    fis, axs = plt.subplots(n, 4, figsize = (30, 4*n))
    for i in range(n):
        df = mouse_pos[starts[i]:ends[i]]
        speed = df.smoothed_speed
        axs[i,0].hist(speed[speed<10], density = True, bins = 100)
        axs[i,0].set_xlim((-0.5,10))
        axs[i,0].set_ylim((0,5))
        axs[i,1].hist(speed[speed>10], density = True, bins = 100)
        axs[i,1].set_xlim((9.5,1000))
        axs[i,1].set_ylim((0,0.06))
        
        acce = df.smoothed_acceleration
        axs[i,2].hist(acce[acce<5], density = True, bins = 100)
        axs[i,2].set_xlim((-0.5,5))
        axs[i,2].set_ylim((0,2))
        axs[i,3].hist(acce[acce>5], density = True, bins = 100)
        axs[i,3].set_xlim((4.5,2000))
        axs[i,3].set_ylim((0, 0.075))
        
        axs[i,0].set_ylabel(starts[i].hour)
    
    plt.savefig(title)
    
def Display_Kinematics_Properties_Along_Time(mouse_pos, title):   
    def Calculate_Properties(dist):
        mean = np.mean(dist)
        variance = np.var(dist)
        skewness = stats.skew(dist)
        kurtosis = stats.kurtosis(dist)
        return mean, variance, skewness, kurtosis

    start = mouse_pos.index[0]
    end = mouse_pos.index[-1]
    starts, ends = [],[]
    while start < end:
        if start.minute != 0:
            end_ = pd.Timestamp(year = start.year, month = start.month, day = start.day, hour = start.hour+1, minute=0, second=0)
        else: 
            end_ = start + pd.Timedelta('1H')

        starts.append(start)
        ends.append(end_)
        start = end_        
    
    Mean_V, Variance_V, Skewness_V, Kurtosis_V = [], [], [], []
    Mean_A, Variance_A, Skewness_A, Kurtosis_A = [], [], [], []
    Hour = []
    n = len(starts)
    
    CR = []
    for i in range(n):
        df = mouse_pos[starts[i]:ends[i]]
        speed = df.smoothed_speed
        mean, variance, skewness, kurtosis = Calculate_Distribution(speed)
        Mean_V.append(mean)
        Variance_V.append(variance)
        Skewness_V.append(skewness)
        Kurtosis_V.append(kurtosis)
        
        acce = df.smoothed_acceleration
        mean, variance, skewness, kurtosis = Calculate_Distribution(acce)
        Mean_A.append(mean)
        Variance_A.append(variance)
        Skewness_A.append(skewness)
        Kurtosis_A.append(kurtosis)
        
        Hour.append(starts[i].hour)
        if starts[i].hour == 7 or starts[i].hour == 19: CR.append(i)
    CR = np.array(CR)
    if starts[CR[0]].hour == 19: CR = np.concatenate((np.array([0]), CR))
    if starts[CR[-1]].hour == 7: CR = np.concatenate((CR, np.array([n-1])))
    
    N = np.arange(n)
    fis, axs = plt.subplots(4, 2, figsize = (30, 20))
    axs[0,0].plot(N, Mean_V)
    axs[0,0].set_ylabel('Mean')
    axs[1,0].plot(N, Variance_V)
    axs[1,0].set_ylabel('Variance')
    axs[2,0].plot(N, Skewness_V)
    axs[2,0].set_ylabel('Skewness')
    axs[3,0].plot(N, Kurtosis_V)
    axs[3,0].set_ylabel('Kurtosis')
    axs[0,1].plot(N, Mean_A)
    axs[1,1].plot(N, Variance_A)
    axs[2,1].plot(N, Skewness_A)
    axs[3,1].plot(N, Kurtosis_A)
    axs[3,0].set_xlabel('Speed')
    axs[3,1].set_xlabel('Acceleration')
    for i in range(4):
        for j in range(2):
            axs[i,j].set_xticks(N[::2], Hour[::2])
            for t in range(0,len(CR),2):
                axs[i,j].axvspan(CR[t],CR[t+1], color='lightblue', alpha=0.5)
    plt.savefig(title)
    
def Display_HMM_TransM(TransM, title)
    annot_array = np.array([[round(item, 3) for item in row] for row in TransM])
    fig, axs = plt.subplots(1,1, figsize=(len(transition_mat)+3, len(transition_mat)+3))
    sns.heatmap(TransM, cmap='binary', ax = axs, square = 'True', cbar = False, annot=annot_array)
    axs.set_title("Learned Transition Matrix") 
    plt.savefig(title)

def Display_HMM_STates_Along_Time(mouse_pos, states, start, end, title):
    mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)
    mouse_pos = mouse_pos[start:end]
    
    grouped = mouse_pos.groupby([pd.Grouper(freq='10S'), 'state']).size()
    prob = grouped.groupby(level=0).apply(lambda g: g / g.sum())
    states_prob = prob.unstack(level=-1).fillna(0)
    states_prob.index = states_prob.index.get_level_values(0)
    
    max_columns = states_prob.idxmax(axis=1)
    states_prob = pd.DataFrame(0, index=states_prob.index, columns=states_prob.columns)
    for row in range(len(states_prob)):
        col = max_columns[row]
        states_prob.at[states_prob.index[row], col] = 1
    
    states_prob['CR'] = 0
    CR_index_1 = states_prob[states_prob.index.hour < 7].index
    CR_index_2 = states_prob[states_prob.index.hour > 19].index
    CR_index = CR_index_1.union(CR_index_2).sort_values()
    states_prob.loc[CR_index, 'CR'] = 1
    
    groups = states_prob['CR'].ne(states_prob['CR'].shift()).cumsum()
    zero_groups = states_prob[states_prob['CR'] == 0].groupby(groups).groups
    zero_groups = list(zero_groups.values())

    START, END = [],[]
    for i in range(len(zero_groups)):
        START.append(zero_groups[i][0])
        END.append(zero_groups[i][-1])


    fig, axs = plt.subplots(N, 1, figsize=(50, 4*N-1))
    for i in range(N):
        states_prob[i].plot(color = color_names[i], ax = axs[i])
        for t in range(len(START)):
            axs[i].axvspan(START[t],END[t], color='lightblue', alpha=0.5)
    
    plt.savefig(title)

def Get_States_Characterized(pellet_delivery = False,state_before_visit = True,start_visit = True,end_visit = True,N=9):
    
    def StartVisit(n=9):    
        STATE, DURATION = [],[]
        for label in LABELS:
            type, mouse = label[0], label[1]
            mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
            states = np.load('../SocialData/HMMStates/' + type + "_" + mouse + "_States.npy", allow_pickle = True)
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

        plt.savefig('../Images/Social_HMM/EnterVisit.png')
        plt.show()
        

        AVE_STATES = []
        for k in np.arange(n):
            index = STATE == k
            states = index*1
            AVE_STATES.append(np.mean(states, axis = 0))
        AVE_STATES = np.array(AVE_STATES)
        fig, axs = plt.subplots(1, 1, figsize=(20, 4))
        for i in range(AVE_STATES.shape[0]):
            color = color_names[i]
            rgba_color = plt.cm.colors.to_rgba(color)
            for j in range(AVE_STATES.shape[1]):
                if np.isnan(AVE_STATES[i, j]):
                    axs.add_patch(plt.Rectangle((j, N-1-i), 0.3, 1, color=plt.cm.colors.to_rgba('black'), alpha=1, linewidth=0))
                else:
                    axs.add_patch(plt.Rectangle((j, N-1-i), 1, 1, color=rgba_color, alpha=AVE_STATES[i, j], linewidth=0))

        axs.set_aspect('auto')
        axs.set_xticks([200])
        axs.set_xticklabels(['Enter'], rotation = 0)
        axs.set_ylabel("States")
        axs.set_yticks(np.arange(N) + 0.5)
        axs.set_yticklabels(np.arange(N-1,-1,-1))
        axs.set_xlim(0, AVE_STATES.shape[1])
        axs.set_ylim(0, AVE_STATES.shape[0])
        plt.savefig('../Images/Social_HMM/EnterVisit_' + 'EachState' + '.png')
        plt.show()

    def EndVisit(n=9):    
        STATES, DURATION = [],[]
        for label in LABELS:
            type, mouse = label[0], label[1]
            mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
            states = np.load('../SocialData/HMMStates/' + type + "_" + mouse + "_States.npy", allow_pickle = True)
            mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)
            
            Visits = pd.read_parquet('../SocialData/VisitData/'  + type + "_" + mouse +'_Visit.parquet', engine='pyarrow')
            Visits = Visits.dropna(subset=['speed'])
            Visits['distance'] = abs(Visits['distance'])
            Visits = Visits[Visits['distance'] >= 0.1]

            for i in range(len(Visits)):
                trigger = Visits.iloc[i]['end'] - pd.Timedelta('10S')
                
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

        plt.savefig('../Images/Social_HMM/EndVisit.png')
        plt.show()
        
        AVE_STATES = []
        for k in np.arange(n):
            index = STATES == k
            states = index*1
            AVE_STATES.append(np.mean(states, axis = 0))
        AVE_STATES = np.array(AVE_STATES)
        fig, axs = plt.subplots(1, 1, figsize=(20, 4))
        for i in range(AVE_STATES.shape[0]):
            color = color_names[i]
            rgba_color = plt.cm.colors.to_rgba(color)
            for j in range(AVE_STATES.shape[1]):
                if np.isnan(AVE_STATES[i, j]):
                    axs.add_patch(plt.Rectangle((j, N-1-i), 0.3, 1, color=plt.cm.colors.to_rgba('black'), alpha=1, linewidth=0))
                else:
                    axs.add_patch(plt.Rectangle((j, N-1-i), 1, 1, color=rgba_color, alpha=AVE_STATES[i, j], linewidth=0))
                    
        axs.set_aspect('auto')
        
        axs.set_xticks([200])
        axs.set_xticklabels(['Leave'], rotation = 0)

        axs.set_ylabel("States")
        #axs.set_yticks([])
        axs.set_yticks(np.arange(N) + 0.5)
        axs.set_yticklabels(np.arange(N-1,-1,-1))
        axs.set_xlim(0, AVE_STATES.shape[1])
        axs.set_ylim(0, AVE_STATES.shape[0])
        plt.savefig('../Images/Social_HMM/EndVisit_' + 'EachState' + '.png')
        plt.show()

    def StateBeforeVisit(n=9):
        STATE, DURATION = [],[]
        for label in LABELS:
            type, mouse = label[0], label[1]
            mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
            states = np.load('../SocialData/HMMStates/' + type + "_" + mouse + "_States.npy", allow_pickle = True)
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
        plt.savefig('../Images/Social_HMM/StateBeforeVisit.png')
        plt.show()
    
    def PelletDeliveries(t = 6, n=9):
    
        Pellets_State = []
        
        for label in LABELS:
            type, mouse = label[0], label[1]
            mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
            states = np.load('../SocialData/HMMStates/' + type + "_" + mouse + "_States.npy", allow_pickle = True)
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
        plt.savefig('../Images/Social_HMM/PelletDelivery.png')
        plt.show()
    
    if pellet_delivery: PelletDeliveries(n=N)
    if state_before_visit: StateBeforeVisit(n=N)
    if start_visit: StartVisit(n=N)
    if end_visit: EndVisit(n=N)

def main():
    
    Mouse = mouse.Mouse(aeon_exp='AEON3', type = 'Pre', mouse = 'BAA-1104045')
    
    '''-------------------------------LDS-------------------------------'''
    Display_LDS_Trace(Mouse.mouse_pos, title = '../Images/Social_LDS/' + Mouse.type + '_' + Mouse.mouse + '.png')
    Display_Kinematics_Distribution_Along_Time(Mouse.mouse_pos, title = '../Images/Social_LDS/Distribution_' + Mouse.type + '_' + Mouse.mouse + '.png')
    Display_Kinematics_Properties_Along_Time(Mouse.mouse_pos,  title = '../Images/Social_LDS/Properties_' + Mouse.type + '_' + Mouse.mouse + '.png')
    
    
    '''-------------------------------HMM-------------------------------'''
    Mouse.hmm.Get_TransM()
    Display_LDS_Trace(Mouse.hmm.TransM, title = '../Images/Social_HMM/TransM_' + Mouse.type + '_' + Mouse.mouse + '.png')
    
    Mouse.hmm.Get_States()
    Display_HMM_STates_Along_Time(Mouse.mouse_pos, Mouse.hmm.states, Mouse.active_chunk[0], Mouse.active_chunk[1], title = '../Images/Social_HMM/EachState/' + Mouse.type + '_' + Mouse.mouse + '.png') 
    
    Mouse.arena.Get_Pellets()
    Mouse.arena.Get_Visits()
    Display_HMM_States_Characterization(Mouse, 
                                        pellet_delivery = True,
                                        state_before_visit = False,
                                        start_visit = True,
                                        end_visit = True,
                                        N=Mouse.hmm.n_state)


if __name__ == "__main__":
        main()
        
        
        
        