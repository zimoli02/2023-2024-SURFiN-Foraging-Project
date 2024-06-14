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
import Functions.HMM as HMM
import Functions.kinematics as kinematics
import Functions.patch as patch
from SSM.ssm.plots import gradient_cmap

parent_dir = current_script_path.parents[2] / 'aeon_mecha' 
sys.path.insert(0, str(parent_dir))
import aeon
import aeon.io.api as api
from aeon.schema.schemas import social02

root = '/ceph/aeon/aeon/data/raw/AEON3/social0.2/'
INFO = pd.read_parquet('../SocialData/INFO3.parquet', engine='pyarrow')
LABELS = [
    ['Pre','BAA-1104045'],
    ['Pre','BAA-1104047'],
    ['Post','BAA-1104045'],
    ['Post','BAA-1104047']
]
nodes_name = ['nose', 'head', 'right_ear', 'left_ear', 'spine1', 'spine2','spine3', 'spine4']
color_names = ['black', "blue", "red", "tan", "green", "brown", "purple", "orange", 'turquoise', "yellow", 'pink', 'darkblue']

def Get_Observations(kinematics_Update, weight_Update, patch_Update, r_Update, bodylength_Update, bodyangle_Update, nose_Update, body_PCA_Update):
    
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

    def FixNan(mouse_pos, column):
        mouse_pos[column] = mouse_pos[column].interpolate()
        mouse_pos[column] = mouse_pos[column].ffill()
        mouse_pos[column] = mouse_pos[column].bfill()
        return mouse_pos


    for i in range(len(LABELS)):
        type, mouse = LABELS[i][0], LABELS[i][1]
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
        if body_PCA_Update:
            pca = PCA(n_components=1)
            mouse_pos['PC1_x'] = pca.fit_transform(data_x.iloc[:, :])
            pca = PCA(n_components=1)
            mouse_pos['PC1_y'] = pca.fit_transform(data_y.iloc[:, :])
            
        mouse_pos.to_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')

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
        np.save('../SocialData/HMMData/LogLikelihood.npy', LogLikelihood)

def Display_Latent_State_Number(N):    
    LogLikelihood = np.load('../SocialData/HMMData/LogLikelihood.npy', allow_pickle=True)
    fig, axs = plt.subplots(1,1,figsize = (10,7))
    for i in range(len(LogLikelihood)):
        loglikelihood = LogLikelihood[i]
        axs.scatter(N, loglikelihood)
        axs.plot(N, loglikelihood, color = 'black', label = i)
    axs.set_xticks(N)
    axs.legend()
    plt.savefig('../Images/Social_HMM/StateNumber.png')
    plt.show()

def Get_Latent_States(id, n, features):
    '''type, mouse = LABELS[id][0], LABELS[id][1]
    mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
    mouse_pos = mouse_pos[pd.Timestamp('2024-02-06 07:00:00.0'):pd.Timestamp('2024-02-07 07:00:00.0')]
    
    obs = np.array(mouse_pos[features])
    hmm, states, transition_mat, lls = HMM.FitHMM(obs, num_states = n, n_iters = 50)
    
    state_mean_speed = hmm.observations.params[0].T[0]
    index = np.argsort(state_mean_speed, -1)     
    
    HMM.PlotTransition(transition_mat[index].T[index].T, title = '../Images/Social_HMM/TransM.png')
    np.save('../SocialData/HMMStates/TransM.npy', transition_mat[index].T[index].T)'''
    
    for label in LABELS:
        type, mouse = label[0], label[1]
        mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
        start = mouse_pos.index[0]
        try:
            start_ = pd.Timestamp(year = start.year, month = start.month, day = start.day+1, hour = 7, minute=0, second=0)
        except ValueError:
            start_ = pd.Timestamp(year = start.year, month = start.month+1, day = 1, hour = 7, minute=0, second=0)
        try:
            end_ = pd.Timestamp(year = start_.year, month = start_.month, day = start_.day+1, hour = 7, minute=0, second=0)
        except ValueError:
            end_ = pd.Timestamp(year = start_.year, month = start_.month+1, day = 1, hour = 7, minute=0, second=0)
        obs_ = np.array(mouse_pos[start_:end_][features])
        hmm, states, transition_mat, lls = HMM.FitHMM(obs_, num_states = n, n_iters = 50)
        state_mean_speed = hmm.observations.params[0].T[0]
        index = np.argsort(state_mean_speed, -1)     
        HMM.PlotTransition(transition_mat[index].T[index].T, title = '../Images/Social_HMM/TransM_' + type + '_' + mouse + '.png')
        np.save('../SocialData/HMMStates/' + type + "_" +mouse + 'TransM.npy', transition_mat[index].T[index].T)
        
        obs = np.array(mouse_pos[features])
        states = hmm.most_likely_states(obs)
        new_values = np.empty_like(states)
        for i, val in enumerate(index): new_values[states == val] = i
        states = new_values
        np.save('../SocialData/HMMStates/' + type + "_" + mouse + "_States.npy", states)

def Display_Latent_States_Features(N):
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
        plt.savefig('../Images/Social_HMM/' + type + "_" +mouse+'.png')
        plt.show()
    
    def CollectData_Single(mouse_pos, N):
        x, y, speed, acce, r, length, angle = [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)]
        for i in range(N):
            x[i] =  mouse_pos['smoothed_position_x'][states==i]
            y[i] = mouse_pos['smoothed_position_y'][states==i]
            speed[i] = mouse_pos['smoothed_speed'][states==i]
            acce[i] = mouse_pos['smoothed_acceleration'][states == i]
            r[i] = mouse_pos['r'][states == i]
            length[i] =mouse_pos['bodylength'][states == i]
            angle[i] = mouse_pos['bodyangle'][states == i]
        return x, y, speed, acce, r, length, angle
    
    def CollectData_All(N, X, Y, SPEED, ACCE,R, LENGTH, ANGLE, x, y, speed, acce, r, length, angle):
        for i in range(N):
            X[i] = np.concatenate([X[i], x[i]])
            Y[i] = np.concatenate([Y[i], y[i]])
            SPEED[i] = np.concatenate([SPEED[i], speed[i]])
            ACCE[i] = np.concatenate([ACCE[i], acce[i]])
            R[i] = np.concatenate([R[i], r[i]])
            LENGTH[i] = np.concatenate([LENGTH[i], length[i]])
            ANGLE[i] = np.concatenate([ANGLE[i], angle[i]])
        return X, Y, SPEED, ACCE,R, LENGTH, ANGLE
    
    def PlotPosition(N, x, y, title):
        fig, axs = plt.subplots(1, N, figsize = (N*8-2,6))
        for i in range(N):
            axs[i].scatter(x[i], y[i], color = color_names[i], s = 2, alpha = 0.2)
            axs[i].set_xlim((100,1400))
            axs[i].set_ylim((-20,1100))
            axs[i].set_title('State' + str(i))
            axs[i].set_xlabel('X')
            axs[i].set_ylabel('Y')
        plt.savefig(title)
        plt.show()
    
    def PlotFeatures(N, DATA, FEATURE, title):
        fig, axs = plt.subplots(len(FEATURE), 1, figsize = (10, len(FEATURE)*7-1))
        for data, i in zip(DATA, range(len(DATA))):
            means = [np.mean(arr) for arr in data]
            var = [np.std(arr)/np.sqrt(len(arr)) for arr in data]
            axs[i].bar(range(N), means, yerr=var, capsize=5)
            axs[i].set_xticks(range(0, N), [str(j) for j in range(N)])
            axs[i].set_ylabel(FEATURE[i])
        plt.savefig(title)
        plt.show()
    
    X, Y, SPEED, ACCE,R, LENGTH, ANGLE = [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)]
    for j in range(len(LABELS)):
        type, mouse = LABELS[j][0], LABELS[j][1]
        mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
        states = np.load('../SocialData/HMMStates/' + type + "_" + mouse + "_States.npy", allow_pickle = True)
        mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)
        
        PlotStates(mouse_pos, N, type, mouse)
        
        x, y, speed, acce, r, length, angle = CollectData_Single(mouse_pos, N)
        PlotPosition(N, x, y, title = '../Images/Social_HMM/Position_' + type + "_" + mouse + '.png')
        PlotFeatures(N, DATA = [speed, acce, r, length, angle], FEATURE = ['SPEED', 'ACCE', 'R', 'LENGTH', 'ANGLE'], title = '../Images/Social_HMM/Feature_' + type + "_" + mouse + '.png')
        
        X, Y, SPEED, ACCE,R, LENGTH, ANGLE = CollectData_All(N, X, Y, SPEED, ACCE,R, LENGTH, ANGLE, x, y, speed, acce, r, length, angle)
    
    PlotPosition(N, X, Y, title = '../Images/Social_HMM/Position_All.png')
    PlotFeatures(N, DATA = [SPEED, ACCE, R, LENGTH, ANGLE], FEATURE = ['SPEED', 'ACCE', 'R', 'LENGTH', 'ANGLE'], title = '../Images/Social_HMM/Feature_All.png')

def Display_Latent_States_Along_Time(N):
    for label in LABELS:
        type, mouse = label[0], label[1]
        mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
        states = np.load('../SocialData/HMMStates/' + type + "_" + mouse + "_States.npy", allow_pickle = True)
        mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)
        
        start = mouse_pos.index[0]
        try:
            start_ = pd.Timestamp(year = start.year, month = start.month, day = start.day+1, hour = 7, minute=0, second=0)
        except ValueError:
            start_ = pd.Timestamp(year = start.year, month = start.month+1, day = 1, hour = 7, minute=0, second=0)
        try:
            end_ = pd.Timestamp(year = start_.year, month = start_.month, day = start_.day+1, hour = 7, minute=0, second=0)
        except ValueError:
            end_ = pd.Timestamp(year = start_.year, month = start_.month+1, day = 1, hour = 7, minute=0, second=0)
        mouse_pos = mouse_pos[start_:end_]
        
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
        
        plt.savefig('../Images/Social_HMM/EachState/' + type + "_" +mouse+'.png')
                
def Calculate_TransM_From_States(N):
    for label in LABELS:
        type, mouse = label[0], label[1]
        mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
        states = np.load('../SocialData/HMMStates/' + type + "_" + mouse + "_States.npy", allow_pickle = True)
        mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)
        
        grouped = mouse_pos.groupby([pd.Grouper(freq='10S'), 'state']).size()
        prob = grouped.groupby(level=0).apply(lambda g: g / g.sum())
        states_prob = prob.unstack(level=-1).fillna(0)
        states_prob.index = states_prob.index.get_level_values(0)
        
        manual_trans_mat = np.zeros((N,N))
        for k in range(1,len(states)): manual_trans_mat[states[k-1]][states[k]] += 1
        for k in range(N): 
            if np.sum(manual_trans_mat[k]) != 0: manual_trans_mat[k] = manual_trans_mat[k]/np.sum(manual_trans_mat[k])

        HMM.PlotTransition(manual_trans_mat, title = '../Images/Social_HMM/'+ type + "_" + mouse +'TransM.png')

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
            '''Visits['distance'] = abs(Visits['distance'])
            Visits = Visits[Visits['distance'] >= 0.1]'''
            
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


    def EndVisit(n=9):    
        STATES, DURATION = [],[]
        for label in LABELS:
            type, mouse = label[0], label[1]
            mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
            states = np.load('../SocialData/HMMStates/' + type + "_" + mouse + "_States.npy", allow_pickle = True)
            mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)
            
            Visits = pd.read_parquet('../SocialData/VisitData/'  + type + "_" + mouse +'_Visit.parquet', engine='pyarrow')
            Visits = Visits.dropna(subset=['speed'])
            '''# The above code snippet is written in Python and performs the following actions:
            Visits['distance'] = abs(Visits['distance'])
            Visits = Visits[Visits['distance'] >= 0.1]'''

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

        plt.savefig('../Images/Social_HMM/EndVisit.png')

        
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

    
    if pellet_delivery: PelletDeliveries(n=N)
    if state_before_visit: StateBeforeVisit(n=N)
    if start_visit: StartVisit(n=N)
    if end_visit: EndVisit(n=N)
    

def main():
    
    features = ['smoothed_speed', 'smoothed_acceleration', 'r','bodylength', 'bodyangle', 'nose']
    #features = ['smoothed_speed', 'smoothed_acceleration', 'r','PC1_x', 'PC1_y']
    
    '''
    Get_Observations(kinematics_Update = False,   
                        weight_Update = False,
                        patch_Update = False,
                        r_Update = False,
                        bodylength_Update = True,
                        bodyangle_Update = True,
                        nose_Update = True,
                        body_PCA_Update = False)
    print('Get_Observations Completed')
    
    
    Get_Latent_State_Number(features, N = np.arange(3,28))
    print('Get_Latent_State_Number Completed')
    
    Display_Latent_State_Number(N = np.arange(3,28))
    print('Display_Latent_State_Number Completed')
    '''
    
    
    '''Get_Latent_States(id=1, n=12, features = features)
    print('Get_Latent_States Completed')
    
    
    Display_Latent_States_Features(N = 12)
    print('Display_Latent_States_Features Completed')
    '''
    
    '''Display_Latent_States_Along_Time(N=12)
    print('Display_Latent_States_Along_Time Completed')
    
    Calculate_TransM_From_States(N=12)  
    print('Calculate_TransM_From_States Completed')'''
    
    Get_States_Characterized(pellet_delivery = False,
                                state_before_visit = False,
                                start_visit = False,
                                end_visit = True,
                                N=12)
    print('Get_States_Characterized Completed')


if __name__ == "__main__":
        main()