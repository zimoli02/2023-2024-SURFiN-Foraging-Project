import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter


import sys
from pathlib import Path
current_script_path = Path(__file__).resolve()

function_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(function_dir))
import Functions.mouse as mouse
from SSM.ssm.plots import gradient_cmap

parent_dir = current_script_path.parents[2] / 'aeon_mecha' 
sys.path.insert(0, str(parent_dir))
import aeon
import aeon.io.api as api
from aeon.io import reader, video
from aeon.schema.schemas import social02

LABELS = [
    ['Pre','BAA-1104045'],
    ['Pre','BAA-1104047'],
    ['Post','BAA-1104045'],
    ['Post','BAA-1104047']
]
nodes_name = ['nose', 'head', 'right_ear', 'left_ear', 'spine1', 'spine2','spine3', 'spine4']
color_names = ['black', "blue", "red", "tan", "green", "brown", "purple", "orange", 'turquoise', "yellow", 'pink', 'darkblue']

def Display_Body_Info(Mouse, property, nodes):
    def n_Cluster(variable):
        n_cluster = {
            'spine1-spine3': 3, 
            'head-spine3': 4, 
            'right_ear-spine3': 5,
            'left_ear-spine3': 5
        }
        return n_cluster[variable]

    def Get_Pose_Frame(variable, value_str, time):
        start, end = time, time + pd.Timedelta("0.2S")
        root = Mouse.root
        video_metadata = aeon.load(root, social02.CameraTop.Video, start=start, end=end)
        video_metadata.index = video_metadata.index.round("20L")
        frames = video.frames(video_metadata)
        first_frame = next(frames)
        cv2.imwrite("../Images/Social_BodyInfo/" + variable + '/frames/' + Mouse_title + '_' + value_str + '.jpg', first_frame)

    
    def DrawBody(data_x, data_y, axs):
        for k in range(len(nodes_name)): 
            axs.scatter(data_x[nodes_name[k]], data_y[nodes_name[k]])
        axs.plot([data_x['nose'],data_x['head']], [data_y['nose'],data_y['head']])
        axs.plot([data_x['left_ear'],data_x['nose']], [data_y['left_ear'],data_y['nose']])
        axs.plot([data_x['nose'],data_x['right_ear']], [data_y['nose'],data_y['right_ear']])
        axs.plot([data_x['left_ear'],data_x['right_ear']], [data_y['left_ear'],data_y['right_ear']])
        axs.plot([data_x['head'],data_x['spine1']], [data_y['head'],data_y['spine1']])
        axs.plot([data_x['spine1'],data_x['spine2']], [data_y['spine1'],data_y['spine2']])
        axs.plot([data_x['spine2'],data_x['spine3']], [data_y['spine2'],data_y['spine3']])
        axs.plot([data_x['spine3'],data_x['spine4']], [data_y['spine3'],data_y['spine4']])
        x_min, y_min = min(np.array(data_x)), min(np.array(data_y))
        axs.set_aspect('equal', 'box')
        axs.set_xlim(x_min-20, x_min+50)
        axs.set_ylim(y_min-20, y_min+50)
        return axs

    def DrawPoses(variable, center, d, axs):
        for j in range(len(center)):
            for i in range(len(data_x)):
                if abs(d[i] - center[j]) < 0.1: 
                    if np.any(np.isnan(np.array(data_x.iloc[i]))): continue
                    axs[j] = DrawBody(data_x.iloc[i],data_y.iloc[i], axs[j])
                    axs[j].set_title(str(round(center[j],2)))
                    Get_Pose_Frame(variable, str(round(center[j],2)), time = times[i])
                    break
        return axs

    def DrawDistance(variable):
        data = mouse_pos[variable].to_numpy()
        kmeans = KMeans(n_clusters=n_Cluster(variable), random_state=0, n_init = 'auto')
        kmeans.fit_predict(data.reshape(-1, 1))
        center = np.sort(kmeans.cluster_centers_.T[0])
        
        fig, axs = plt.subplots(1,len(center), figsize = (len(center)*5,4))
        axs = DrawPoses(variable,center, data, axs)
        plt.savefig('../Images/Social_BodyInfo/'+ variable + '/' + Mouse_title + '.png')
        plt.show()

    Mouse_title = Mouse.type + '_' + Mouse.mouse
    times = Mouse.mouse_pos.index
    mouse_pos = Mouse.mouse_pos
    
    variable = nodes[0]
    for i in range(1, len(nodes)): variable = variable + '-' + nodes[i]
    
    data_x = Mouse.body_data_x
    data_y = Mouse.body_data_y
    
    DrawDistance(variable)

    print('Display_Body_Info for variable ' + variable + ' Completed')
    
def Display_Body_Info_Comparison(Mice, Bodylength = True, Bodyangle = True, Nose = True):
    def Compare_Pose_between_Animals(pattern, data, cluster = 5):
        N = len(Mice)
        fig, axs = plt.subplots(N, 1, figsize = (8, N*4))
        for i in range(N):
            d = data[i].reshape(-1, 1)
            kmeans = KMeans(n_clusters=cluster, random_state=0)
            kmeans.fit(d)
            center = np.sort(kmeans.cluster_centers_.T[0])
            
            axs[i].hist(data[i], bins = 100, color = 'blue')
            for j in range(cluster):
                axs[i].axvline(x = center[j], color = 'red', linestyle = '--')
            axs[i].set_ylabel(LABELS[i][0] + "-" + LABELS[i][1])
            
            if pattern == 'BodyLength': axs[i].set_xlim((-1, 50))
            if pattern == 'BodyAngle': axs[i].set_xlim((-1, 180))
            if pattern == 'Nose': axs[i].set_xlim((0, 45))
        axs[N-1].set_xlabel(pattern)
        plt.savefig('../Images/Social_' + pattern + '/Summary.png')

    
    BodyLength, BodyAngle, NoseActivity = [], [], []
    for Mouse in Mice:
        bodylength = Mouse.body_info.bodylength
        bodyangle = Mouse.body_info.bodyangle
        nose = Mouse.body_info.nose
        BodyLength.append(bodylength)
        BodyAngle.append(bodyangle)
        NoseActivity.append(nose)
    
    if Bodylength: Compare_Pose_between_Animals(pattern = 'BodyLength', data = BodyLength, cluster = 5)
    if Bodyangle: Compare_Pose_between_Animals(pattern = 'BodyAngle', data = BodyAngle, cluster = 5)
    if Nose: Compare_Pose_between_Animals(pattern = 'Nose', data = NoseActivity, cluster = 3)
    print('Display_Body_Info_Comparison Completed')

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
    print('Display_LDS_Trace Completed')
    
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
    print('Display_Kinematics_Distribution_Along_Time Completed')
    
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
        mean, variance, skewness, kurtosis = Calculate_Properties(speed)
        Mean_V.append(mean)
        Variance_V.append(variance)
        Skewness_V.append(skewness)
        Kurtosis_V.append(kurtosis)
        
        acce = df.smoothed_acceleration
        mean, variance, skewness, kurtosis = Calculate_Properties(acce)
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
    print('Display_Kinematics_Properties_Along_Time Completed')
    
def Display_HMM_TransM(TransM, title):
    annot_array = np.array([[round(item, 3) for item in row] for row in TransM])
    fig, axs = plt.subplots(1,1, figsize=(len(TransM)+3, len(TransM)+3))
    sns.heatmap(TransM, cmap='binary', ax = axs, square = 'True', cbar = False, annot=annot_array)
    axs.set_title("Learned Transition Matrix") 
    plt.savefig(title)
    print('Display_HMM_TransM Completed')

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

    N = max(states) + 1
    fig, axs = plt.subplots(N, 1, figsize=(50, 4*N-1))
    for i in range(N):
        states_prob[i].plot(color = color_names[i], ax = axs[i])
        for t in range(len(START)):
            axs[i].axvspan(START[t],END[t], color='lightblue', alpha=0.5)
    
    plt.savefig(title)
    
    print('Display_HMM_STates_Along_Time Completed')

def Display_HMM_States_Feature(Mouse, title):
    def CollectData_Single(mouse_pos, N):
        x, y, speed, acce, r, spine1_spine3, head_spine3, right_ear_spine3, left_ear_spine3 = [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)]
        for i in range(N):
            x[i] =  mouse_pos['smoothed_position_x'][states==i]
            y[i] = mouse_pos['smoothed_position_y'][states==i]
            speed[i] = mouse_pos['smoothed_speed'][states==i]
            acce[i] = mouse_pos['smoothed_acceleration'][states == i]
            r[i] = mouse_pos['r'][states == i]
            spine1_spine3[i] = mouse_pos['spine1-spine3'][states == i]
            head_spine3[i] = mouse_pos['head-spine3'][states == i]
            right_ear_spine3[i] = mouse_pos['right_ear-spine3'][states == i]
            left_ear_spine3[i] = mouse_pos['left_ear-spine3'][states == i]
        return x, y, speed, acce, r, spine1_spine3, head_spine3, right_ear_spine3, left_ear_spine3

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
    
    def PlotFeatures(N, DATA, FEATURE, title):
        fig, axs = plt.subplots(len(FEATURE), 1, figsize = (10, len(FEATURE)*7-1))
        for data, i in zip(DATA, range(len(DATA))):
            means = [np.mean(arr) for arr in data]
            var = [np.std(arr)/np.sqrt(len(arr)) for arr in data]
            axs[i].bar(range(N), means, yerr=var, capsize=5)
            axs[i].set_xticks(range(0, N), [str(j) for j in range(N)])
            axs[i].set_ylabel(FEATURE[i])
        plt.savefig(title)
    
    Mouse_title = Mouse.type + '_' + Mouse.mouse
    mouse_pos = Mouse.mouse_pos
    states = Mouse.hmm.states
    N = Mouse.hmm.n_state
    mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)

    x, y, speed, acce, r, spine1_spine3, head_spine3, right_ear_spine3, left_ear_spine3 = CollectData_Single(mouse_pos, N)
    PlotPosition(N, x, y, title = '../Images/Social_HMM/Position/' + Mouse_title + '.png')
    PlotFeatures(N, DATA = [speed, acce, r, spine1_spine3, head_spine3, right_ear_spine3, left_ear_spine3], FEATURE = ['SPEED', 'ACCE', 'R', 'Spine1-3', 'Head-Spine3', 'RightE-Spine3', 'LeftE-Spine3'], title = '../Images/Social_HMM/Feature/' + Mouse_title + '.png')
    print('Display_HMM_States_Feature Completed')

def Display_HMM_States_Characterization(Mouse, pellet_delivery = False, state_before_visit = True, start_visit = True, end_visit = True, enter_arena = True):

    Mouse_title = Mouse.type + '_' + Mouse.mouse
    mouse_pos = Mouse.mouse_pos
    states = Mouse.hmm.states
    N = Mouse.hmm.n_state
    Pellets = Mouse.arena.pellets
    Visits = Mouse.arena.visits
    Entry = Mouse.arena.entry 
    
    mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)
    Visits = Visits.dropna(subset=['speed'])
    
    colors = sns.xkcd_palette(color_names[0:N])
    cmap = gradient_cmap(colors)
    
    def StartVisit(N):        
        STATE, DURATION = [], []    
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
        
        fig, axs = plt.subplots(1, 1, figsize=(10, 16))
        sns.heatmap(STATE, cmap=cmap, ax=axs, vmin=0, vmax = N-1, cbar = True)
        axs.set_aspect('auto')
        
        axs.set_xticks([200])
        axs.set_xticklabels(['Enter'], rotation = 0)

        axs.set_ylabel("Visits")
        axs.set_yticks([])

        plt.savefig('../Images/Social_HMM/EnterVisit/' + Mouse_title + '.png')
        

        AVE_STATES = []
        for k in np.arange(N):
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
        plt.savefig('../Images/Social_HMM/EnterVisit_EachState/' + Mouse_title + '.png')

    def EndVisit(N):            
        STATES, DURATION = [],[]
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

        fig, axs = plt.subplots(1, 1, figsize=(10, 16))
        sns.heatmap(STATES,cmap=cmap, ax=axs, vmin=0, vmax = N-1, cbar = True)
        axs.set_aspect('auto')
        
        axs.set_xticks([200])
        axs.set_xticklabels(['Leave'], rotation = 0)

        axs.set_ylabel("Visits")
        axs.set_yticks([])

        plt.savefig('../Images/Social_HMM/EndVisit/' + Mouse_title + '.png')

        
        AVE_STATES = []
        for k in np.arange(N):
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
        plt.savefig('../Images/Social_HMM/EndVisit_EachState/' + Mouse_title + '.png')

    def StateBeforeVisit(N):
        STATE, DURATION = [],[]

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
        plt.savefig('../Images/Social_HMM/StateBeforeVisit/' + Mouse_title + '.png')

    def PelletDeliveries(N, t = 6):
        Pellets_State = []
        for i in range(len(Pellets)):
            trigger = Pellets.index[i]
            
            latest_valid_index = mouse_pos.loc[trigger - pd.Timedelta('6S'):trigger, 'state'].index
            latest_valid_state = mouse_pos.loc[latest_valid_index, ['state']].values.reshape(-1)
            if len(latest_valid_state) >= 50: latest_valid_state  = latest_valid_state[-50:]
            
            next_valid_index = mouse_pos.loc[trigger:trigger + pd.Timedelta('6S'), 'state'].index
            next_valid_state = mouse_pos.loc[next_valid_index, ['state']].values.reshape(-1)
            if len(next_valid_state) >= 50: next_valid_state  = next_valid_state[:50]
            
            state = np.concatenate((latest_valid_state, np.array([np.nan]), next_valid_state))
            
            if len(state) == 101: Pellets_State.append(state)

        fig, axs = plt.subplots(1, 1, figsize=(6, 16))
        sns.heatmap(Pellets_State,cmap=cmap, ax=axs, vmin=0, vmax = N-1, cbar = True)
        axs.set_aspect('auto')
        axs.set_xticks([50])
        axs.set_xticklabels(['Pellet'], rotation = 0)
        axs.set_ylabel("Pellet Deliveries")
        axs.set_yticks([])
        plt.savefig('../Images/Social_HMM/PelletDelivery/' + Mouse_title + '.png')

    def EnterArena():
        STATES_day, STATES_night = [],  []
        for i in range(len(Entry)):
            trigger = Entry[i]
            
            latest_valid_index = mouse_pos.loc[trigger - pd.Timedelta('11S'):trigger, 'state'].index
            latest_valid_state = mouse_pos.loc[latest_valid_index, ['state']].values.reshape(-1)
            if len(latest_valid_state) >= 100: latest_valid_state  = latest_valid_state[-100:]
            
            next_valid_index = mouse_pos.loc[trigger:trigger + pd.Timedelta('11S'), 'state'].index
            next_valid_state = mouse_pos.loc[next_valid_index, ['state']].values.reshape(-1)
            if len(next_valid_state) >= 100: next_valid_state  = next_valid_state[:100]
            
            state = np.concatenate((latest_valid_state, np.array([np.nan]), next_valid_state))
            
            if len(state) == 201: 
                if trigger.hour < 7 or trigger.hour > 19: STATES_night.append(state)
                else: STATES_day.append(state)
        STATES = np.vstack((np.array(STATES_day), np.array(STATES_night)))
        fig, axs = plt.subplots(1, 1, figsize=(10, 16))
        sns.heatmap(STATES,cmap=cmap, ax=axs, vmin=0, vmax = N-1, cbar = True)
        axs.set_aspect('auto')
        
        axs.set_xticks([100])
        axs.set_xticklabels(['Enter'], rotation = 0)

        axs.set_ylabel("Visits")
        axs.set_yticks([])

        plt.savefig('../Images/Social_HMM/EnterArena/' + Mouse_title + '.png')

        
        AVE_STATES = []
        for k in np.arange(N):
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
        
        axs.set_xticks([100])
        axs.set_xticklabels(['Enter'], rotation = 0)

        axs.set_ylabel("States")
        axs.set_yticks(np.arange(N) + 0.5)
        axs.set_yticklabels(np.arange(N-1,-1,-1))
        axs.set_xlim(0, AVE_STATES.shape[1])
        axs.set_ylim(0, AVE_STATES.shape[0])
        plt.savefig('../Images/Social_HMM/EnterArena_EachState/' + Mouse_title + '.png')
    
    if pellet_delivery: PelletDeliveries(N)
    if state_before_visit: StateBeforeVisit(N)
    if start_visit: StartVisit(N)
    if end_visit: EndVisit(N)
    if enter_arena: EnterArena()
    print('Display_HMM_States_Characterization Completed')

def main():
    for label in LABELS:
        type_name, mouse_name = label[0], label[1]
        Mouse = mouse.Mouse(aeon_exp='AEON3', type = type_name, mouse = mouse_name)
        
        '''-------------------------------BODY-------------------------------'''
        NODES = [['head', 'spine3'],['spine1', 'spine3'],['left_ear', 'spine3'],['right_ear', 'spine3']]
        for nodes in NODES:
            Mouse.Add_Body_Info_to_mouse_pos(property = 'distance', nodes = nodes)
            #Display_Body_Info(Mouse, property = 'distance', nodes = nodes)
        '''
        Display_Body_Info_Comparison(Mice = [Mouse], Bodylength = True, Bodyangle = True, Nose = True)
        '''
    
        '''-------------------------------LDS-------------------------------'''
        '''
        Display_LDS_Trace(Mouse.mouse_pos, title = '../Images/Social_LDS/' + Mouse.type + '_' + Mouse.mouse + '.png')
        Display_Kinematics_Distribution_Along_Time(Mouse.mouse_pos, title = '../Images/Social_LDS/Distribution_' + Mouse.type + '_' + Mouse.mouse + '.png')
        Display_Kinematics_Properties_Along_Time(Mouse.mouse_pos,  title = '../Images/Social_LDS/Properties_' + Mouse.type + '_' + Mouse.mouse + '.png')
        '''
        
        '''-------------------------------HMM-------------------------------'''
        #Mouse.hmm.Fit_Model(n_state = 12, feature = 'Kinematics_and_Body')
        Mouse.hmm.n_state = 12
        Mouse.hmm.feature = 'Kinematics_and_Body'
        Mouse.hmm.Get_TransM(n_state = 12, feature = 'Kinematics_and_Body')
        Display_HMM_TransM(Mouse.hmm.TransM, title = '../Images/Social_HMM/TransM/' + Mouse.type + '_' + Mouse.mouse + '.png')
        
        Mouse.hmm.Get_States()
        Display_HMM_STates_Along_Time(Mouse.mouse_pos, Mouse.hmm.states, Mouse.active_chunk[0], Mouse.active_chunk[1], title = '../Images/Social_HMM/State/' + Mouse.type + '_' + Mouse.mouse + '.png') 
        
        Mouse.Run_Visits()
        Display_HMM_States_Feature(Mouse, title = '../Images/Social_HMM/Feature/' + Mouse.type + '_' + Mouse.mouse + '.png')
        Display_HMM_States_Characterization(Mouse, 
                                            pellet_delivery = True,
                                            state_before_visit = False,
                                            start_visit = True,
                                            end_visit = True,
                                            enter_arena = True)

if __name__ == "__main__":
        main()
        
        
        
        