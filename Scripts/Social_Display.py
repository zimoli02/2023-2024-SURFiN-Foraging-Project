import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
import tensorflow as tf
from tensorflow import keras

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
color_names = [
    'black', 'blue', 'red', 'tan', 'green', 'brown', 
    'purple', 'orange', 'turquoise', 'yellow', 'pink', 
    'darkblue', 'lime', 'cyan', 'magenta', 'gold', 
    'navy', 'maroon', 'teal', 'grey'
]

'''-------------------------------BODY-------------------------------'''
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
        clusters = kmeans.fit_predict(data.reshape(-1, 1))
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
    
def Display_Body_Info_Characterization(Mouse, NODES, pellet_delivery = True, start_visit = True, end_visit = True, enter_arena = True):
    def n_Cluster(variable):
        n_cluster = {
            'spine1-spine3': 3, 
            'head-spine3': 4, 
            'right_ear-spine3': 5,
            'left_ear-spine3': 5
        }
        return n_cluster[variable]
    
    def find_column_modes(arr):
        modes, _ = stats.mode(arr)
        return modes[0]

    def Characterize_Timepoints(event_name, Events, left_seconds, right_seconds, file_name):
        left_period = pd.Timedelta(str(left_seconds+1) + 'S')
        right_period = pd.Timedelta(str(right_seconds+1) + 'S')
        
        colors = sns.xkcd_palette(color_names[0:5])
        cmap = gradient_cmap(colors)
        CLUSTERS, VARIABLES = [], []
        for nodes in NODES:
            variable = nodes[0]
            for i in range(1, len(nodes)): variable = variable + '-' + nodes[i]
            
            data = mouse_pos[variable].to_numpy()
            kmeans = KMeans(n_clusters=n_Cluster(variable), random_state=0, n_init = 'auto')
            clusters = kmeans.fit_predict(data.reshape(-1, 1))
            center = np.sort(kmeans.cluster_centers_.T[0])
            
            index = np.argsort(center, -1)     

            new_values = np.empty_like(clusters)
            for i, val in enumerate(index): new_values[clusters == val] = i
            clusters = new_values
            mouse_pos['cluster'] = pd.Series(clusters, index = mouse_pos.index)

            clusters = []
            for i in range(len(Events)):
                if event_name!= None: trigger = Events.iloc[i][event_name]
                else: trigger = Events[i]
                
                latest_valid_index = mouse_pos.loc[trigger - left_period:trigger, 'cluster'].index
                latest_valid_state = mouse_pos.loc[latest_valid_index, ['cluster']].values.reshape(-1)
                if len(latest_valid_state) >= 10*left_seconds: latest_valid_state  = latest_valid_state[-10*left_seconds:]
                
                next_valid_index = mouse_pos.loc[trigger:trigger + right_period, 'cluster'].index
                next_valid_state = mouse_pos.loc[next_valid_index, ['cluster']].values.reshape(-1)
                if len(next_valid_state) >= 10*right_seconds: next_valid_state  = next_valid_state[:10*right_seconds]
                
                cluster = np.concatenate((latest_valid_state, np.array([np.nan]), next_valid_state))
                
                if len(cluster) == 10*(left_seconds + right_seconds) + 1: 
                    clusters.append(cluster)
                    
            clusters = find_column_modes(clusters)
            CLUSTERS.append(clusters)
            VARIABLES.append(variable)
        
        CLUSTERS = np.array(CLUSTERS)
        fig, axs = plt.subplots(1, 1, figsize=(10, 4))
        sns.heatmap(CLUSTERS, cmap=cmap, ax=axs, vmin=0, vmax = 5-1, cbar = True)
        axs.set_aspect('auto')
        axs.set_xticks([10*left_seconds])
        axs.set_xticklabels([event_name], rotation = 0)
        axs.set_ylabel("Events")
        axs.set_yticks([])
        plt.savefig('../Images/Social_BodyInfo/' + file_name + '/' + Mouse_title + '.png')

        
    Mouse_title = Mouse.type + '_' + Mouse.mouse
    mouse_pos = Mouse.mouse_pos
    
    Mouse.Run_Visits()
    Pellets = Mouse.arena.pellets.index
    Visits = Mouse.arena.visits
    Entry = Mouse.arena.entry
    
    if pellet_delivery: Characterize_Timepoints(event_name = None, Events = Pellets, left_seconds = 20, right_seconds = 10, file_name = 'PelletDelivery')
    if start_visit: Characterize_Timepoints(event_name = 'start', Events = Visits, left_seconds = 20, right_seconds = 10, file_name = 'EnterVisit')
    if end_visit: Characterize_Timepoints(event_name = 'end', Events = Visits, left_seconds = 20, right_seconds = 10, file_name = 'EndVisit')
    if enter_arena: Characterize_Timepoints(event_name = None, Events = Entry, left_seconds = 20, right_seconds = 10, file_name = 'EnterArena')
    print('Display_Body_Info_Characterization Completed')

'''-------------------------------LDS-------------------------------'''
def Display_LDS_Trace(Mouse, file_path):
    Mouse_title = Mouse.type + '_' + Mouse.mouse
    mouse_pos = Mouse.mouse_pos
    
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
    plt.savefig(file_path + Mouse_title + '.png')
    print('Display_LDS_Trace Completed')
    
def Display_Kinematics_Distribution_Along_Time(Mouse, file_path):
    Mouse_title = Mouse.type + '_' + Mouse.mouse
    mouse_pos = Mouse.mouse_pos
    start, end = mouse_pos.index[0], mouse_pos.index[-1]
    starts, ends = [],[]
    while start < end:
        end_ = start + pd.Timedelta('4H')

        starts.append(start)
        ends.append(end_)
        start = end_ + pd.Timedelta('1S')

    n = len(starts)
    fig, axs = plt.subplots(n, 4, figsize = (30, 4*n))
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
    plt.savefig(fie_path + Mouse_title+'.png')
    
    print('Display_Kinematics_Distribution_Along_Time Completed')
    
def Display_Kinematics_Properties_Along_Time(Mouse, file_path):   
    def Calculate_Properties(dist):
        mean = np.mean(dist)
        variance = np.var(dist)
        skewness = stats.skew(dist)
        kurtosis = stats.kurtosis(dist)
        return mean, variance, skewness, kurtosis

    Mouse_title = Mouse_title = Mouse.type + '_' + Mouse.mouse
    mouse_pos = Mouse.mouse_pos
    start, end = mouse_pos.index[0], mouse_pos.index[-1]
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

'''-------------------------------HMM-------------------------------'''    
def Display_HMM_TransM(Mouse, file_path):
    Mouse_title = Mouse.type + '_' + Mouse.mouse
    TransM = Mouse.hmm.TransM
    annot_array = np.array([[round(item, 3) for item in row] for row in TransM])
    fig, axs = plt.subplots(1,1, figsize=(len(TransM)+3, len(TransM)+3))
    sns.heatmap(TransM, cmap='binary', ax = axs, square = 'True', cbar = False, annot=annot_array)
    axs.set_title("Learned Transition Matrix") 
    plt.savefig( file_path + Mouse_title + '.png')
    print('Display_HMM_TransM Completed')

def Display_HMM_States_Along_Time(Mouse, file_path):
    Mouse_title = Mouse.type + '_' + Mouse.mouse
    mouse_pos = Mouse.mouse_pos
    N = Mouse.hmm.n_state
    states = Mouse.hmm.states
    start, end = Mouse.active_chunk[0], Mouse.active_chunk[1]
    
    mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)
    mouse_pos = mouse_pos[start:end]
    
    states_prob = Mouse.hmm.Process_States.State_Dominance(mouse_pos, time_seconds = 10)
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
    plt.savefig(file_path + Mouse_title + '.png')
    
    print('Display_HMM_STates_Along_Time Completed')

def Display_HMM_States_Feature(Mouse, file_path):
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
    PlotPosition(N, x, y, title = file_path + 'Position/' + Mouse_title + '.png')
    PlotFeatures(N, DATA = [speed, acce, r, spine1_spine3, head_spine3, right_ear_spine3, left_ear_spine3], FEATURE = ['SPEED', 'ACCE', 'R', 'Spine1-3', 'Head-Spine3', 'RightE-Spine3', 'LeftE-Spine3'], title = file_path + 'Feature/' + Mouse_title + '.png')
    print('Display_HMM_States_Feature Completed')

def Display_HMM_States_Characterization(Mouse, pellet_delivery = False, start_visit = True, end_visit = True, enter_arena = True):
    def Characterize_Timepoints(event_name, Events, left_seconds, right_seconds, file_name):
        left_period = pd.Timedelta(str(left_seconds+1) + 'S')
        right_period = pd.Timedelta(str(right_seconds+1) + 'S')
        STATES = []
        for i in range(len(Events)):
            trigger = Events[i]
            
            latest_valid_index = mouse_pos.loc[trigger - left_period:trigger, 'state'].index
            latest_valid_state = mouse_pos.loc[latest_valid_index, ['state']].values.reshape(-1)
            if len(latest_valid_state) >= 10*left_seconds: latest_valid_state  = latest_valid_state[-10*left_seconds:]
            
            next_valid_index = mouse_pos.loc[trigger:trigger + right_period, 'state'].index
            next_valid_state = mouse_pos.loc[next_valid_index, ['state']].values.reshape(-1)
            if len(next_valid_state) >= 10*right_seconds: next_valid_state  = next_valid_state[:10*right_seconds]
            
            state = np.concatenate((latest_valid_state, np.array([np.nan]), next_valid_state))
            
            if len(state) == 10*(left_seconds + right_seconds) + 1: 
                STATES.append(state)
        STATES = np.array(STATES)
        
        fig, axs = plt.subplots(1, 1, figsize=(10, 16))
        sns.heatmap(STATES,cmap=cmap, ax=axs, vmin=0, vmax = N-1, cbar = True)
        axs.set_aspect('auto')
        axs.set_xticks([10*left_seconds])
        axs.set_xticklabels([event_name], rotation = 0)
        axs.set_ylabel("Events")
        axs.set_yticks([])
        plt.savefig('../Images/Social_HMM/' + file_name + '/' + Mouse_title + '.png')

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
        axs.set_xticks([10*left_seconds])
        axs.set_xticklabels([event_name], rotation = 0)
        axs.set_ylabel("States")
        axs.set_yticks(np.arange(N) + 0.5)
        axs.set_yticklabels(np.arange(N-1,-1,-1))
        axs.set_xlim(0, AVE_STATES.shape[1])
        axs.set_ylim(0, AVE_STATES.shape[0])
        plt.savefig('../Images/Social_HMM/' + file_name + '_EachState/' + Mouse_title + '.png')
    
    Mouse_title = Mouse.type + '_' + Mouse.mouse
    mouse_pos = Mouse.mouse_pos
    states = Mouse.hmm.states
    N = Mouse.hmm.n_state
    
    Pellets = Mouse.arena.pellets.index
    Visits = Mouse.arena.visits.dropna(subset=['speed'])
    Starts = Visits['start']
    Ends = Visits['end']
    Entry = Mouse.arena.entry 
    
    mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)
    
    colors = sns.xkcd_palette(color_names[0:N])
    cmap = gradient_cmap(colors)
    
    if pellet_delivery: Characterize_Timepoints('Pellet Delivery', Pellets, left_seconds = 5, right_seconds = 5, file_name = 'PelletDelivery')
    if start_visit: Characterize_Timepoints('Move Wheel', Starts, left_seconds = 20, right_seconds = 5, file_name = 'EnterVisit')
    if end_visit: Characterize_Timepoints('Leave Wheel', Ends, left_seconds = 20, right_seconds = 5, file_name = 'EndVisit')
    if enter_arena: Characterize_Timepoints('Enter Arena', Entry, left_seconds = 10, right_seconds = 10, file_name = 'EnterArena')
    print('Display_HMM_States_Characterization Completed')

def Display_HMM_States_Predicting_Behavior_Gaussian(Mouse, pellet_delivery = True, start_visit = True, end_visit = True, enter_arena = True):

    def Calculate_Probability_Curve(characterized_states_curve = [], time_shifts = [], means = [], variances = []):
        T = len(characterized_states_curve[0])
        P = np.ones(T)
        p_ = []
        
        for i in range(len(characterized_states_curve)):
            p = np.zeros(T)
            characterized_state, time_shift, mean, variance = characterized_states_curve[i], time_shifts[i], means[i], variances[i]
            if time_shift < 0:
                p[-time_shift:] = np.exp(-((characterized_state[:time_shift] - mean) ** 2) / (2 * variance ** 2))
            else:
                p[:-time_shift] = np.exp(-((characterized_state[time_shift:] - mean) ** 2) / (2 * variance ** 2))
            p_.append(p)
    
        for p in p_: P *= p
        return P

    def Calculate_All_Count_Curve (mouse_pos):
        states_prob = Mouse.hmm.Process_States.State_Probability(mouse_pos, time_seconds = 10)
        for i in range(N): mouse_pos.loc[mouse_pos.index, 'State' + str(i)] = states_prob[i].to_numpy()
        return mouse_pos
    
    def Summarize(event_name, Events):
        mouse_pos_ = mouse_pos.copy()[Active_Chunk[0]:Active_Chunk[1]]

        COUNT_CURVES = [[] for _ in range(N)]
        COUNT_CURVES_NAMES = []
        for i in range(len(Events)):
            if event_name!= None: trigger = Events.iloc[i][event_name]
            else: trigger = Events[i]
            
            if trigger < Active_Chunk[0] or trigger > Active_Chunk[1]: continue
            
            latest_valid_index = mouse_pos_.loc[trigger - pd.Timedelta('6S'):trigger].index
            next_valid_index = mouse_pos_.loc[trigger:trigger + pd.Timedelta('6S')].index
            
            for j in range(N):
                latest_valid_state = mouse_pos_.loc[latest_valid_index, ['State' + str(j)]].values.reshape(-1)
                if len(latest_valid_state) >= 50: latest_valid_state  = latest_valid_state[-50:]
                next_valid_state = mouse_pos_.loc[next_valid_index, ['State' + str(j)]].values.reshape(-1)
                if len(next_valid_state) >= 50: next_valid_state  = next_valid_state[:50]            
                prob = np.concatenate((latest_valid_state, next_valid_state))
                if len(prob) == 100: COUNT_CURVES[j].append(prob)
        
        COUNT_CURVES_MAX = []
        for i in range(N): 
            COUNT_CURVES[i] = np.mean(np.array(COUNT_CURVES[i]), axis = 0)
            COUNT_CURVES_MAX.append(np.max(COUNT_CURVES[i]))
            COUNT_CURVES_NAMES.append('State' + str(i))
        
        Peaks_index = np.argsort(COUNT_CURVES_MAX)
        if COUNT_CURVES_MAX[Peaks_index[-4]] > 0.4: threshold = COUNT_CURVES_MAX[Peaks_index[-4]]
        elif COUNT_CURVES_MAX[Peaks_index[-3]] < 0.4: threshold = COUNT_CURVES_MAX[Peaks_index[-3]]
        else: threshold = 0.4
        
        characterized_states, characterized_states_names, time_shifts = [], [], []
        for i in range(N):
            if np.max(COUNT_CURVES[i]) > threshold:
                characterized_states.append(i)
                characterized_states_names.append('State'+str(i))
                max_index = np.argsort(COUNT_CURVES[i], -1)[0]
                time_shifts.append(max_index - 50 + 5)
        return characterized_states, characterized_states_names, time_shifts
        
    def Predict(event_name, Events, file_name):
        characterized_states, characterized_states_names, time_shifts = Summarize(event_name, Events)
        means = [1, 1,1,1,1,1, 1, 1,1,1,1,1]
        variances = [0.1, 0.1, 0.1,0.1,0.1, 0.1, 0.1, 0.1, 0.1,0.1,0.1, 0.1]
        
        if len(characterized_states) == 0:
            return 'No states characterized'

        characterized_states_curve = []
        for i in range(len(characterized_states)):
            characterized_states_curve.append(mouse_pos.loc[mouse_pos.index, characterized_states_names[i]].to_numpy())
        probability_curve = Calculate_Probability_Curve(characterized_states_curve, time_shifts = time_shifts, means = means, variances = variances)
        mouse_pos.loc[mouse_pos.index, 'prob'] = probability_curve

        fig, axs = plt.subplots(1, 1, figsize=(10, 6))
        T = np.arange(-50, 50, 1)
        PROB= []
        COUNT_CURVES = [[] for _ in range(len(characterized_states_curve))]
        for i in range(len(Events)):
            trigger = Events[i]
            if trigger < Predicting_Chunk_start: continue
            
            latest_valid_index = mouse_pos.loc[trigger - pd.Timedelta('6S'):trigger, 'prob'].index
            next_valid_index = mouse_pos.loc[trigger:trigger + pd.Timedelta('6S'), 'prob'].index
            
            latest_valid_state = mouse_pos.loc[latest_valid_index, ['prob']].values.reshape(-1)
            if len(latest_valid_state) >= 50: latest_valid_state  = latest_valid_state[-50:]
            next_valid_state = mouse_pos.loc[next_valid_index, ['prob']].values.reshape(-1)
            if len(next_valid_state) >= 50: next_valid_state  = next_valid_state[:50]            
            prob = np.concatenate((latest_valid_state, next_valid_state))
            if len(prob) == 100: PROB.append(prob)
            
            for j in range(len(characterized_states_curve)):
                latest_valid_state = mouse_pos.loc[latest_valid_index, [characterized_states_names[j]]].values.reshape(-1)
                if len(latest_valid_state) >= 50: latest_valid_state  = latest_valid_state[-50:]
                next_valid_state = mouse_pos.loc[next_valid_index, [characterized_states_names[j]]].values.reshape(-1)
                if len(next_valid_state) >= 50: next_valid_state  = next_valid_state[:50]            
                prob = np.concatenate((latest_valid_state, next_valid_state))
                if len(prob) == 100: COUNT_CURVES[j].append(prob)
                
        
        for i in range(len(characterized_states_curve)):
            axs.plot(T, np.mean(np.array(COUNT_CURVES[i]), axis = 0), label = characterized_states_names[i], linestyle = '--')
        axs.axvline(x = 0, color = 'red')
        axs.legend()
        axs_ = axs.twinx()
        axs_.plot(T, np.mean(np.array(PROB), axis = 0), color = 'black', label = 'Pred.')
        axs_.legend()
        plt.savefig('../Images/Social_HMM/' + file_name + '/' + Mouse_title + '_Prediction.png')
        return 'Predicton for ' + file_name + ' Completed'
    
    Mouse_title = Mouse.type + '_' + Mouse.mouse
    mouse_pos = Mouse.mouse_pos
    N = Mouse.hmm.n_state
    states = Mouse.hmm.states
    mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)

    Pellets = Mouse.arena.pellets.index
    Visits = Mouse.arena.visits
    Entry = Mouse.arena.entry 
    Starts = Visits['start']
    Ends = Visits['end']

    Active_Chunk = Mouse.active_chunk
    Predicting_Chunk_start = Active_Chunk[1] + pd.Timedelta('1S')
    
    mouse_pos = Calculate_All_Count_Curve(mouse_pos)

    if pellet_delivery: 
        result = Predict(event_name = 'Pellet Delivery', Events = Pellets, file_name = 'PelletDelivery')
        print(result)
    if start_visit: 
        result = Predict(event_name = 'Move Wheel', Events = Starts, file_name = 'EnterVisit')
        print(result)
    if end_visit: 
        result = Predict(event_name = 'Leave Wheel', Events = Ends, file_name = 'EndVisit')
        print(result)
    if enter_arena: 
        result = Predict(event_name = 'Enter Arena', Events = Entry, file_name = 'EnterArena')
        print(result)
    
    print('Display_HMM_States_Predicting_Behavior_Gaussian Completed')

def Display_HMM_States_Predicting_Behavior_MLP(Mouse, pellet_delivery = True, start_visit = True, end_visit = True, enter_arena = True):
    def Train(event_name, Events):
        mouse_pos_ = mouse_pos.copy()[Active_Chunk[0]:Active_Chunk[1]]

        train_inputs, train_outputs = [], []
        for i in range(len(Events)):
            trigger = Events[i]
            if trigger < Active_Chunk[0] or trigger > Active_Chunk[1]: continue
            
            latest_valid_index = mouse_pos_.loc[trigger - pd.Timedelta('4S'):trigger].index
            latest_valid_state = mouse_pos_.loc[latest_valid_index, ['state']].values.reshape(-1)
            if len(latest_valid_state) >= 30: 
                latest_valid_state  = latest_valid_state[-30:]
                train_inputs.append(latest_valid_state)
                train_outputs.append(1)
            
            latest_valid_index = mouse_pos_.loc[trigger - pd.Timedelta('7S'):trigger- pd.Timedelta('3S')].index
            latest_valid_state = mouse_pos_.loc[latest_valid_index, ['state']].values.reshape(-1)
            if len(latest_valid_state) >= 30: 
                latest_valid_state  = latest_valid_state[-30:]
                train_inputs.append(latest_valid_state)
                train_outputs.append(0.5)
            
            next_valid_index = mouse_pos_.loc[trigger:trigger + pd.Timedelta('3S')].index
            next_valid_state = mouse_pos_.loc[next_valid_index, ['state']].values.reshape(-1)
            if len(next_valid_state) >= 30: 
                next_valid_state  = next_valid_state[:30]   
                train_inputs.append(next_valid_state)
                train_outputs.append(-0.3)       
                
            next_valid_index = mouse_pos_.loc[trigger + pd.Timedelta('3S'):trigger + pd.Timedelta('7S')].index
            next_valid_state = mouse_pos_.loc[next_valid_index, ['state']].values.reshape(-1)
            if len(next_valid_state) >= 30: 
                next_valid_state  = next_valid_state[:30]   
                train_inputs.append(next_valid_state)
                train_outputs.append(0) 
                
        train_inputs, train_outputs = np.array(train_inputs), np.array(train_outputs)
        train_inputs = train_inputs.astype(np.int32)
        train_outputs = train_outputs.astype(np.float32)
        input_shape = (len(train_inputs[0]),)
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(4, activation='softmax')
        ])
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        model.fit(train_inputs, train_outputs, epochs=50, batch_size=32)
        
        return model
            
    def Predict(event_name, Events, file_name):
        model = Train(event_name, Events)
        mouse_pos_ = mouse_pos.copy()[Predicting_Chunk_start:]
        test_indices = mouse_pos_.index
        test_inputs = []
        state = mouse_pos_['state'].to_numpy()
        for i in range(0, len(state)-30, 10):
            states = state[i: i+30]
            test_inputs.append(states)
        test_inputs = np.array(test_inputs)
        test_inputs = test_inputs.astype(np.int32)
        
        predictions = model.predict((np.array(test_inputs)))
        predicted_labels = np.argmax(predictions, axis=1)
        output_values = [-0.3, 0, 0.5, 1]
        predicted_outputs = [output_values[label] for label in predicted_labels]
        
        mouse_pos_.loc[test_indices, 'pred'] = 0
        for i in range(0,len(test_indices)-30, 10):
            mouse_pos_.loc[test_indices[i+30], 'pred'] = predicted_outputs[int(i/10)]
        
        fig, axs = plt.subplots(1, 1, figsize=(30, 4))
        mouse_pos_.pred.plot(ax = axs, color = 'black')
        for i in range(len(Events)):
            trigger = Events[i]
            if trigger < Predicting_Chunk_start: continue
            axs.axvline(x = trigger, color = 'red')
        plt.savefig('../Images/Social_HMM/' + file_name + '/' + Mouse_title + '_Prediction_MLP.png')
        return 'Predicton for ' + file_name + ' Completed'
        
    Mouse_title = Mouse.type + '_' + Mouse.mouse
    mouse_pos = Mouse.mouse_pos
    N = Mouse.hmm.n_state
    states = Mouse.hmm.states
    mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)

    Pellets = Mouse.arena.pellets.index
    Visits = Mouse.arena.visits
    Entry = Mouse.arena.entry 
    Starts = Visits['start']
    Ends = Visitis['end']

    Active_Chunk = Mouse.active_chunk
    Predicting_Chunk_start = Active_Chunk[1] + pd.Timedelta('1S')
    
    if pellet_delivery: 
        result = Predict(event_name = 'Pellet Delivery', Events = Pellets, file_name = 'PelletDelivery')
        print(result)
    if start_visit: 
        result = Predict(event_name = 'Move Wheel', Events = Starts, file_name = 'EnterVisit')
        print(result)
    if end_visit: 
        result = Predict(event_name = 'Leave Wheel', Events = Ends, file_name = 'EndVisit')
        print(result)
    if enter_arena: 
        result = Predict(event_name = 'Enter Arena', Events = Entry, file_name = 'EnterArena')
        print(result)
    
    print('Display_HMM_States_Predicting_Behavior_MLP Completed')

def main():
    for label in LABELS:
        type_name, mouse_name = label[0], label[1]
        print('Start Processing: ', type_name, "-", mouse_name)
        
        Mouse = mouse.Mouse(aeon_exp='AEON3', type = type_name, mouse = mouse_name)
        
        '''-------------------------------BODY-------------------------------'''
        
        NODES = [['head', 'spine3'],['spine1', 'spine3'],['left_ear', 'spine3'],['right_ear', 'spine3']]
        for nodes in NODES:
            Mouse.Add_Body_Info_to_mouse_pos(property = 'distance', nodes = nodes)
            Display_Body_Info(Mouse, property = 'distance', nodes = nodes)
        
        Display_Body_Info_Characterization(Mouse, NODES,
                                            pellet_delivery = True,
                                            start_visit = True,
                                            end_visit = True,
                                            enter_arena = True)

    
        '''-------------------------------LDS-------------------------------'''
        
        Display_LDS_Trace(Mouse, file_path = '../Images/Social_LDS/')
        Display_Kinematics_Distribution_Along_Time(Mouse, file_path = '../Images/Social_LDS/Distribution_')
        Display_Kinematics_Properties_Along_Time(Mouse,  file_path = '../Images/Social_LDS/Properties_')
        
        
        '''-------------------------------HMM-------------------------------'''
        #Mouse.hmm.Fit_Model(n_state = 12, feature = 'Kinematics_and_Body')
        Mouse.hmm.n_state = 20
        Mouse.hmm.feature = 'Kinematics_and_Body'
        Mouse.hmm.Get_TransM(n_state = 20, feature = 'Kinematics_and_Body')
        Mouse.hmm.Get_States()
        Mouse.Run_Visits()
        
        Display_HMM_TransM(Mouse, file_path = '../Images/Social_HMM/TransM/')
        Display_HMM_States_Along_Time(Mouse, file_path = '../Images/Social_HMM/State/') 
        Display_HMM_States_Feature(Mouse, file_path = '../Images/Social_HMM/')
        Display_HMM_States_Characterization(Mouse, 
                                            pellet_delivery = True,
                                            start_visit = True,
                                            end_visit = True,
                                            enter_arena = True)
        Display_HMM_States_Predicting_Behavior_Gaussian(Mouse,
                                                        pellet_delivery = True,
                                                        start_visit = True,
                                                        end_visit = True,
                                                        enter_arena = True)
        Display_HMM_States_Predicting_Behavior_MLP(Mouse,
                                                    pellet_delivery = True,
                                                    start_visit = True,
                                                    end_visit = True,
                                                    enter_arena = True)
        


if __name__ == "__main__":
        main()
        
        
        
        