import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

import statsmodels.api as sm
import scipy.stats as stats
import random
from scipy.special import factorial
from dtaidistance import dtw

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from collections import Counter

import sys
from pathlib import Path
current_script_path = Path(__file__).resolve()

functions_dir = current_script_path.parents[1] / 'Functions'
sys.path.insert(0, str(functions_dir))
import mouse as mouse

ssm_dir = current_script_path.parents[2] / 'SSM'
sys.path.insert(0, str(ssm_dir))
import ssm as ssm
from ssm.plots import gradient_cmap

aeon_mecha_dir = current_script_path.parents[2] / 'aeon_mecha' 
sys.path.insert(0, str(aeon_mecha_dir))
import aeon
import aeon.io.api as api
from aeon.io import reader, video
from aeon.schema.schemas import social02

LABELS = [  
    ['AEON3', 'Pre','BAA-1104045'],
    ['AEON3', 'Pre','BAA-1104047'],  
    ['AEON3', 'Post','BAA-1104045'],     
    ['AEON3', 'Post','BAA-1104047'],   
    ['AEON4', 'Pre','BAA-1104048'],
    ['AEON4', 'Pre','BAA-1104049'],
    ['AEON4', 'Post','BAA-1104048'],
    ['AEON4', 'Post','BAA-1104049'] 
]

'''
    ['AEON3', 'Pre','BAA-1104045'],
    ['AEON3', 'Pre','BAA-1104047'],  
    ['AEON3', 'Post','BAA-1104045'],     
    ['AEON3', 'Post','BAA-1104047'],   
    ['AEON4', 'Pre','BAA-1104048'],
    ['AEON4', 'Pre','BAA-1104049'],
    ['AEON4', 'Post','BAA-1104048'],
    ['AEON4', 'Post','BAA-1104049'] 
'''

nodes_name = ['nose', 'head', 'right_ear', 'left_ear', 'spine1', 'spine2','spine3', 'spine4']
color_names = [
    'black', 'blue', 'red', 'tan', 'green', 'brown', 
    'purple', 'orange', 'magenta', 'olive', 'pink', 
    'darkblue', 'lime', 'cyan', 'turquoise', 'gold', 
    'navy', 'maroon', 'teal', 'grey']

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
            
            clusters = Mouse.hmm.process_states.Event_Triggering(mouse_pos, Events, left_seconds, right_seconds, 'cluster')
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
    states = Mouse.hmm.states
    N = Mouse.hmm.n_state
    
    Pellets = Mouse.arena.pellets.index
    Visits = Mouse.arena.visits.dropna(subset=['speed'])
    Starts = Visits['start']
    Ends = Visits['end']
    Entry = Mouse.arena.entry 
    
    if pellet_delivery: Characterize_Timepoints(event_name = 'Pellet Delivery', Events = Pellets, left_seconds = 20, right_seconds = 10, file_name = 'PelletDelivery')
    if start_visit: Characterize_Timepoints(event_name = 'Move Wheel', Events = Starts, left_seconds = 20, right_seconds = 10, file_name = 'EnterVisit')
    if end_visit: Characterize_Timepoints(event_name = 'End Wheel', Events = Ends, left_seconds = 20, right_seconds = 10, file_name = 'EndVisit')
    if enter_arena: Characterize_Timepoints(event_name = 'Enter Arena', Events = Entry, left_seconds = 20, right_seconds = 10, file_name = 'EnterArena')
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
    plt.tight_layout()
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
    plt.savefig(file_path + Mouse_title+'.png')
    
    print('Display_Kinematics_Distribution_Along_Time Completed')
    
def Display_Kinematics_Properties_Along_Time(Mouse, file_path):   
    Mouse_title = Mouse.type + '_' + Mouse.mouse
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
    
    Full_Hour = np.array([i % 24 for i in range(24*4)])
    Full_Sequence = np.arange(4*24)
    Full_CR = np.array([[7 + 24*i, 19 + 24*i] for i in range(4)])
    
    Mean_V, Variance_V, Mean_A, Variance_A = np.full(24*4, np.nan),np.full(24*4, np.nan),np.full(24*4, np.nan),np.full(24*4, np.nan)

    day = 0
    for i in range(len(starts)):
        hour = starts[i].hour
        if hour == 0: day += 1
        
        index_in_full_sequence = day*24 + hour
        
        df = mouse_pos[starts[i]:ends[i]]
        if len(df) == 0: continue
        
        speed = df.smoothed_speed
        Mean_V[index_in_full_sequence] = np.mean(speed)
        Variance_V[index_in_full_sequence] = np.var(speed)
            
        acce = df.smoothed_acceleration
        Mean_A[index_in_full_sequence] = np.mean(acce)
        Variance_A[index_in_full_sequence] = np.var(speed)

    fig, axs = plt.subplots(2, 2, figsize = (24, 16))
    axs[0][0].plot(Full_Sequence, Mean_V)
    axs[0][0].set_ylim(0,75)
    axs[0][0].set_ylabel('Mean', fontsize = 40)
    axs[1][0].plot(Full_Sequence, Variance_V)
    axs[1][0].set_ylim(0,12000)
    axs[1][0].set_ylabel('Variance', fontsize = 40)
    axs[1][0].set_xlabel('Hours', fontsize = 40)
    axs[0][0].set_title('Speed', fontsize = 45)

    axs[0][1].plot(Full_Sequence, Mean_A)
    axs[0][1].set_ylabel('Mean', fontsize = 40)
    axs[0][1].set_ylim(0,150)
    axs[1][1].plot(Full_Sequence, Variance_A)
    axs[1][1].set_ylim(0,100000)
    axs[1][1].set_ylabel('Variance', fontsize = 40)
    axs[1][1].set_xlabel('Hours', fontsize = 40)
    axs[0][1].set_title('Acceleration', fontsize = 45)

    for i in range(2):
        for j in range(2):
            #axs[i][j].legend(loc = 'upper right', fontsize = 25)
            axs[i][j].set_xticks(Full_Sequence[::6], Full_Hour[::6])
            axs[i][j].tick_params(axis='both', which='major', labelsize=25)
            for t in range(len(Full_CR)):
                axs[i][j].axvspan(Full_CR[t][0],Full_CR[t][1], color='lightblue', alpha=0.5)
            axs[i][j].spines['top'].set_visible(False)
            axs[i][j].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(file_path + Mouse_title+'.png')
    print('Display_Kinematics_Properties_Along_Time Completed')

'''-------------------------------HMM-------------------------------'''   
def Display_Model_Selection(Mouse, N, file_path):
    Mouse_title = Mouse.type + '_' + Mouse.mouse
    try:
        Loglikelihood = np.load('../../SocialData/HMMStates/Loglikelihood_' + Mouse_title + '.npy', allow_pickle=True)

    except FileNotFoundError:
        Loglikelihood = []
        for n in N:
            Mouse.hmm.Fit_Model_without_Saving(n_state = n, feature = 'Kinematics_and_Body')
            Loglikelihood.append(Mouse.hmm.loglikelihood)
            print('End Inference for n = ', str(n))
        Loglikelihood = np.array(Loglikelihood)
        np.save('../SocialData/HMMStates/Loglikelihood_' + Mouse_title + '.npy', Loglikelihood)
    
    points = 24*60*60*10 
    Loglikelihood = Loglikelihood/points
    df = Loglikelihood[1:] - Loglikelihood[:-1]
    
    fig, axs = plt.subplots(1,2,figsize = (20,8))
    axs[0].scatter(N, Loglikelihood)
    axs[0].plot(N, Loglikelihood)
    axs[0].set_xticks(N)
    axs[1].scatter(N[1:], df)
    axs[1].plot(N[1:], df)
    axs[1].set_xticks(N[1:])
    for i in range(2):
        axs[i].axvline(x=10, color = 'red', linestyle = "--")
        axs[i].set_xlabel('State Number', fontsize = 20)
        axs[i].tick_params(axis='both', which='major', labelsize=12)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
    axs[0].set_ylabel('Log Likelihood per Point', fontsize = 20)
    axs[1].set_ylabel('$\Delta$Log Likelihood per Point', fontsize = 20)
    plt.tight_layout()

def Display_HMM_TransM(Mouse, file_path, exclude_diag = False):
    Mouse_title = Mouse.type + '_' + Mouse.mouse
    
    if not exclude_diag:
        TransM = Mouse.hmm.TransM
    else:
        TransM = np.zeros((10,10))
        states = Mouse.hmm.states
        for i in range(len(states)-1):
            if states[i+1] != states[i]:
                TransM[states[i]][states[i+1]] += 1
        for i in range(10):
            TransM[i] = TransM[i]/np.sum(TransM[i])
    
    annot_array = np.array([[round(item, 3) for item in row] for row in TransM])
    labels = ['$S_{' + str(i+1) + '}$' for i in range(len(TransM))]
    
    fig, axs = plt.subplots(1,1, figsize=(10,8))
    sns.heatmap(TransM, cmap='RdBu', ax = axs, square = 'True', cbar = True, annot=annot_array, annot_kws={'size': 14})
    axs.set_title("Transition Matrix", fontsize = 25)
    axs.set_xticklabels(labels)
    axs.set_yticklabels(labels, rotation = 0)
    axs.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()
    plt.savefig( file_path + Mouse_title + '.png')
    
    print('Display_HMM_TransM Completed')
    
def Display_HMM_KLMatrix(Mouse, file_path):
    Mouse_title = Mouse.type + '_' + Mouse.mouse
    
    kl_matrix = np.log10(Mouse.hmm.kl_divergence + 1)
    
    annot_array = np.array([[round(item, 1) for item in row] for row in kl_matrix])
    labels = ['$S_{' + str(i+1) + '}$' for i in range(len(kl_matrix))]
    
    fig, axs = plt.subplots(1,1, figsize=(10,8))
    sns.heatmap(kl_matrix, cmap='RdBu', ax = axs, square = 'True', cbar = True, vmin = 0, vmax = 6, annot=annot_array, annot_kws={'size': 14})
    axs.set_title("Kl-Divergence Matrix", fontsize = 25)
    axs.set_xticklabels(labels)
    axs.set_yticklabels(labels, rotation = 0)
    axs.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()
    plt.savefig(file_path + Mouse_title + '.png')
    
    print('Display_HMM_KLMatrix Completed')

def Display_HMM_States_Along_Time(Mouse, file_path):
    Mouse_title = Mouse.type + '_' + Mouse.mouse
    mouse_pos = Mouse.mouse_pos
    N = Mouse.hmm.n_state
    states = Mouse.hmm.states
    start, end = Mouse.active_chunk[0], Mouse.active_chunk[1]
    
    mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)
    mouse_pos = mouse_pos[start:end]
    
    states_prob = Mouse.hmm.process_states.State_Dominance(mouse_pos, time_seconds = 10)
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

def Display_HMM_States_Duration_Along_Time(Mouse, file_path):
    Mouse_title = Mouse.type + '_' + Mouse.mouse
    mouse_pos = Mouse.mouse_pos
    N = Mouse.hmm.n_state
    
    states = Mouse.hmm.states
    mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)
    
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
        
    StateFreq = np.zeros((N, len(starts)))
    Hour = []
    for i in range(len(starts)):
        Hour.append(starts[i].hour)
        mouse_pos_ = mouse_pos[starts[i]:ends[i]]
        states = mouse_pos_.state.to_numpy()
        
        count = np.zeros(N)
        for num in states:
            count[num] += 1
        
        if len(states) != 0: count = count/len(states)
        
        for j in range(N):
            StateFreq[j][i] = count[j]

    row_labels = ['S '+str(i+1) for i in range(N)]
    column_labels = [str(Hour[i]) for i in range(len(Hour)) ]
    fig, axs = plt.subplots(1,1,figsize = (20, 12))
    sns.heatmap(StateFreq, cmap='RdBu', ax = axs, square = 'True', cbar = False)
    axs.set_title("State Frequency", fontsize = 25)
    axs.set_xlabel('Hour', fontsize = 16)
    axs.set_ylabel('State', fontsize = 16)
    axs.set_xticks(np.arange(0.5, len(column_labels)+0.5, 1))
    axs.set_xticklabels(column_labels, rotation = 0)
    axs.set_yticks(np.arange(0.5,len(row_labels)+0.5,1))
    axs.set_yticklabels(row_labels, rotation = 0)
    axs.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(file_path + Mouse_title + '_Frequency.png')


def Display_HMM_States_Feature(Mouse, file_path):
    def CollectData_Single(mouse_pos):
        x, y, r= [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)]
        for i in range(N):
            x[i] = mouse_pos['smoothed_position_x'][states==i]
            y[i] = mouse_pos['smoothed_position_y'][states==i]
            r[i] = mouse_pos['r'][states == i]
        return x, y, r
        
    def PlotPosition(x, y, title):
        fig, axs = plt.subplots(1, N, figsize = (N*8-2,6))
        for i in range(N):
            axs[i].scatter(x[i], y[i], color = color_names[i], s = 0.001, alpha = 1)
            axs[i].set_xlim((100,1400))
            axs[i].set_ylim((-20,1100))
            axs[i].set_title('State' + str(i+1), fontsize = 20)
            axs[i].set_xlabel('X (px)', fontsize = 16)
            axs[i].set_ylabel('Y (px)', fontsize = 16)
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
        #plt.tight_layout()
        plt.savefig(title)
    
    def PlotFeatures(feature, FEATURE, title):
        scale = 2e-3
        indices = [Mouse.hmm.features.index(element) for element in feature]
        
        N_zoom_state = 7
        fig, axs = plt.subplots(3, 2, figsize = (N + N_zoom_state, 3*7-1), gridspec_kw={'width_ratios': [N, N_zoom_state]})
        for i in range(3):
            if i > 1: scale = 2
            axs[i][0].bar(range(N), Params[0][indices[i]][:N]*scale, yerr=Params[1][indices[i]][:N]* scale**2, capsize=14)
            axs[i][0].set_xticks(range(0, N), [str(j+1) for j in range(N)])
            axs[i][1].bar(range(N_zoom_state), Params[0][indices[i]][:N_zoom_state]*scale, yerr=Params[1][indices[i]][:N_zoom_state]* scale**2, capsize=14)
            axs[i][1].set_xticks(range(0, N_zoom_state), [str(j+1) for j in range(N_zoom_state)])
        for i in range(3):
            for j in range(2):
                axs[i][j].set_ylabel(FEATURE[i], fontsize = 35)
                axs[i][j].spines['top'].set_visible(False)
                axs[i][j].spines['right'].set_visible(False)
                axs[i][j].tick_params(axis='both', which='major', labelsize=30)
                axs[2][j].set_xlabel('State', fontsize = 40)
        plt.tight_layout()
        plt.savefig(title)        
        
    
    Mouse_title = Mouse.type + '_' + Mouse.mouse
    mouse_pos = Mouse.mouse_pos
    states = Mouse.hmm.states
    N = Mouse.hmm.n_state
    Params = Mouse.hmm.parameters
    mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)

    x, y, r, = CollectData_Single(mouse_pos)
    PlotPosition(x, y, title = file_path + 'Position/' + Mouse_title + '.png')
    PlotFeatures(['smoothed_speed', 'smoothed_acceleration', 'head-spine3'], FEATURE = ['SPEED (m/s)', 'ACCELERATION (m/s$^2$)', 'BODY LENGTH (mm)'], title = file_path + 'Feature/' + Mouse_title + '.png')
    print('Display_HMM_States_Feature Completed')

def Display_HMM_States_Characterization(Mouse, pellet_delivery = False, start_visit = True, end_visit = True, enter_arena = True, file_path = '../Images/Social_HMM/'):
        
    def EventHeatmap(STATES, sequence, left_seconds, right_seconds, file_name):
        fig, axs = plt.subplots(2, 1, figsize=(10, 10.5),gridspec_kw={'height_ratios': [10, 0.5]})
        sns.heatmap(STATES,cmap=cmap, ax=axs[0], vmin=0, vmax = Mouse.hmm.n_state-1, cbar = False)
        sns.heatmap(np.array([sequence]),cmap=cmap, ax=axs[1], vmin=0, vmax = Mouse.hmm.n_state-1, cbar = False)
        for i in range(2):
            axs[i].set_aspect('auto')
            axs[i].set_xticks([])
            axs[i].set_xticklabels([], rotation = 0)
            axs[i].set_ylabel('')
            axs[i].set_yticks([])
            
        flags = [0]
        for i in range(1,len(sequence)-1):
            if i-1 == left_seconds*10 +1 or i == left_seconds*10 + 1 or i+1 == left_seconds*10 + 1: continue
            if sequence[i] != sequence[i-1]: flags.append(i)
        flags.append(len(sequence))
        for i in range(len(flags)-1):
            state = sequence[flags[i]]
            state_length = flags[i+1] - flags[i]
            if state_length < 3 :continue
            center_position = (flags[i+1] + flags[i])/2
            axs[1].text(center_position, 0.5, f'{int(state+1)}', 
                        ha='center', va='center', fontsize=24, 
                        fontweight='bold', color='white',
                        transform=axs[1].transData)

        fig.suptitle(file_name, fontsize=36, y=0.98)
        axs[0].text(0, 1.0, f't=-{left_seconds}s', fontsize=20, ha='left', va='bottom', transform=axs[0].transAxes)
        axs[0].text(1.0*left_seconds/(left_seconds + right_seconds + 0.1), 1.0, f't=0s', fontsize=20, ha='center', va='bottom', transform=axs[0].transAxes)
        axs[0].text(1, 1.0, f't=+{right_seconds}s', fontsize=20, ha='right', va='bottom', transform=axs[0].transAxes)
        plt.tight_layout()
        plt.savefig(file_path + file_name + '/' + Mouse_title + '.png')
        
    def EventHEatmap_EachState(event_name, STATES,left_seconds, file_name):
        AVE_STATES = []
        for k in np.arange(N):
            index = STATES == k
            states = index*1
            AVE_STATES.append(np.mean(states, axis = 0))
        AVE_STATES = np.array(AVE_STATES)
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        for i in range(AVE_STATES.shape[0]):
            color = color_names[i]
            rgba_color = plt.cm.colors.to_rgba(color)
            for j in range(AVE_STATES.shape[1]):
                if np.isnan(AVE_STATES[i, j]):
                    axs.add_patch(plt.Rectangle((j, N-1-i), 0.2, 1, color=plt.cm.colors.to_rgba('black'), alpha=1, linewidth=0))
                else:
                    axs.add_patch(plt.Rectangle((j, N-1-i), 1, 1, color=rgba_color, alpha=AVE_STATES[i, j], linewidth=0))
        axs.set_aspect('auto')
        axs.set_xticks([10*left_seconds])
        axs.set_xticklabels([event_name], rotation = 0)
        axs.set_ylabel("States")
        axs.set_yticks(np.arange(N) + 0.5)
        axs.set_yticklabels(np.arange(N,0,-1))
        axs.set_xlim(0, AVE_STATES.shape[1])
        axs.set_ylim(0, AVE_STATES.shape[0])
        plt.savefig(file_path + file_name + '_EachState/' + Mouse_title + '.png')
        
    def EventPosition(event_name, STATES, X, Y, file_name):
        fig, axs = plt.subplots(1, 1, figsize=(20, 20))
        mask = ~np.isnan(STATES[0])
        for i in range(len(STATES)):
            colors = np.array(color_names)[STATES[i][mask].astype(int)]
            x = X[i][mask]
            y = Y[i][mask]
            dx = np.diff(x)
            dy = np.diff(y)
            axs.quiver(x[:-1], y[:-1], dx, dy, color=colors[:-1], 
                    angles='xy', scale_units='xy', scale=1.1, 
                    width=0.001, headwidth=3, headlength=3)
        axs.set_aspect('equal')
        axs.set_xlim((100,1400))
        axs.set_ylim((-20,1100))
        plt.savefig(file_path + file_name + '/' + Mouse_title + '_Position.png')
        
        if event_name != 'Enter Arena':
            fig, axs = plt.subplots(1, 3, figsize=(30, 10))
            for i in range(len(STATES)):
                colors = np.array(color_names)[STATES[i][mask].astype(int)]
                x = X[i][mask]
                y = Y[i][mask]
                dx = np.diff(x)
                dy = np.diff(y)
                for j in range(len(Mouse.arena.patch)):
                    axs[j].quiver(x[:-1], y[:-1], dx, dy, color=colors[:-1], 
                        angles='xy', scale_units='xy', scale=1.1, 
                        width=0.001, headwidth=4, headlength=4)
            
            patch_r = Mouse.arena.patch_r
            for i in range(len(Mouse.arena.patch)):
                patch = Mouse.arena.patch[i]
                axs[i].set_aspect('equal')
                patch_ox, patch_oy = Mouse.arena.patch_location[patch][0], Mouse.arena.patch_location[patch][1]
                x_min, x_max, y_min, y_max = patch_ox-patch_r, patch_ox+patch_r, patch_oy-patch_r, patch_oy+patch_r
                axs[i].set_xlim(x_min, x_max)
                axs[i].set_ylim(y_min, y_max)
                axs[i].set_title(patch)

            plt.tight_layout()
            plt.savefig(file_path + file_name + '/' + Mouse_title + '_Position_EachPatch.png')
    
    def EventSequence(STATES, sequence, STATES_Random, left_seconds, right_seconds, file_name):
        def test(pre, post):
            valid = ~np.isnan(pre)
            pre_valid = pre[valid]
            valid = ~np.isnan(post)
            post_valid = post[valid]
            
            statistic, p_value = stats.ks_2samp(pre_valid, post_valid)
            return statistic, p_value
        
        STATES_ = np.delete(STATES, left_seconds*10, axis = 1)
        sequence_ = np.delete(np.array([np.array(sequence)]), left_seconds*10, axis = 1)
        STATES_Random_ = np.delete(STATES_Random, left_seconds*10, axis = 1)
        
        
        Distance = []
        for s in range(len(STATES_)):
            distance = []
            for j in range(len(STATES_[s])):
                distance.append(kl_matrix[int(STATES_[s][j])][int(sequence_[0][j])])
            Distance.append(np.array(distance))
        Distance = np.array(Distance)

        sec = int(len(STATES)/4)
        fig, axs = plt.subplots(1,1,figsize = (10, 8))
        axs.plot(np.arange(len(STATES_[0])),np.mean(Distance[:sec], axis = 0) + 1, color = 'brown', alpha = 1, label = 'Early')
        axs.plot(np.arange(len(STATES_[0])),np.mean(Distance[sec:-sec], axis = 0) + 1, color = 'green', alpha = 1, label = 'Mid')
        axs.plot(np.arange(len(STATES_[0])),np.mean(Distance[-sec:], axis = 0) + 1, color = 'blue', alpha = 1, label = 'Late')
        #axs.plot(np.arange(len(STATES_[0])),sequence_[0]+1, color = 'black', alpha = 1, label = 'Dominant')
        axs.axvline(x = left_seconds*10, linestyle = '--', color = 'red')
        axs.legend(fontsize = 20)
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        axs.set_ylabel('(Log) Distance', fontsize = 30)
        axs.set_xlabel('Time (s)', fontsize = 30)
        axs.set_xticks([0,left_seconds*10,(left_seconds+right_seconds)*10-1])
        axs.set_xticklabels(['-' + str(left_seconds),'0','+' + str(right_seconds)])
        axs.tick_params(axis='both', which='major', labelsize=20)
        plt.tight_layout()
        plt.savefig(file_path + file_name + '/' + Mouse_title + 'Sequence.png')
        
        CORRE, CORRE_Shuffle = [], []
        start = int((left_seconds-1.5) * 10)
        end = int((left_seconds+1.5) * 10)
        dominant_sequence = sequence_[0][start:end] + 1
        for i in range(len(STATES)):
            state = STATES_[i][start:end] + 1
            state_ = STATES_Random_[i][start:end] + 1
            CORRE.append(Mouse.hmm.process_states.Compare_Sequence(state, dominant_sequence, kl_matrix))
            CORRE_Shuffle.append(Mouse.hmm.process_states.Compare_Sequence(state_, dominant_sequence, kl_matrix))
        np.save('../../SocialData/HMMStates/Corre_'+ file_name + '_' + Mouse_title + ".npy", CORRE)
        
        fig, axs = plt.subplots(1,1,figsize = (3, 10))
        axs.plot(CORRE, np.arange(len(STATES),0,-1),color = 'red')
        axs.set_xlim((0,50))
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        axs.set_ylabel('#Events', fontsize = 20)
        axs.set_xlabel('Distance', fontsize = 22)
        axs.set_xticks([0,25,50])
        axs.set_xticklabels(['0','25','50'])
        axs.set_yticks([0,len(CORRE)])
        axs.set_yticklabels(['Late','Early'])
        axs.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        plt.savefig(file_path + file_name + '/' + Mouse_title + 'SequenceCorre.png')


        statistic, p_value = test(np.array(CORRE), np.array(CORRE_Shuffle))
        fig, axs = plt.subplots(1, 1, figsize=(12, 3))
        axs.hist(CORRE, bins = 30, color = 'red', alpha = 0.5, label = 'Original')
        axs.hist(CORRE_Shuffle,bins = 30,  color = 'blue', alpha = 0.5, label = 'Shuffled')
        axs.plot([],[], color = 'white', label = 'p value = ' + str(round(p_value, 3)))
        axs.legend(loc = 'upper right', fontsize = 20)
        #axs.set_title(category, fontsize=30)
        #axs.set_xticks(np.arange(-1, 1.1, 0.5))
        axs.set_ylabel('Count.', fontsize=20)
        axs.set_xlabel('Distance', fontsize=20)
        #axs[i].set_ylim(0, 1.0)  # Adjust if needed
        axs.tick_params(axis='both', which='major', labelsize=18)
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(file_path + file_name + '/' + Mouse_title + 'SequenceCorre_Validation.png')
    
    def Characterize_Timepoints(event_name, Events, left_seconds, right_seconds, file_name):
        STATES = Mouse.hmm.process_states.Event_Triggering(mouse_pos, Events, left_seconds, right_seconds, 'state')
        sequence = Mouse.hmm.process_states.Find_Event_Sequence(STATES)
        X = Mouse.hmm.process_states.Event_Triggering(mouse_pos, Events, left_seconds, right_seconds, 'smoothed_position_x')
        Y = Mouse.hmm.process_states.Event_Triggering(mouse_pos, Events, left_seconds, right_seconds, 'smoothed_position_y')
        
        
        #EventHeatmap(STATES, sequence, left_seconds, right_seconds, file_name)
        #EventPosition(event_name, STATES, X, Y, file_name)
        #EventHEatmap_EachState(event_name, STATES,left_seconds, file_name)
        
        idx = random.sample(range(len(Mouse.mouse_pos)-100), len(STATES))
        random_events = Mouse.mouse_pos.index[idx]
        STATES_Random = Mouse.hmm.process_states.Event_Triggering(Mouse.mouse_pos, random_events, left_seconds, right_seconds, 'state')
        EventSequence(STATES, sequence, STATES_Random, left_seconds, right_seconds, file_name)
        
    
    Mouse_title = Mouse.type + '_' + Mouse.mouse
    mouse_pos = Mouse.mouse_pos
    N = Mouse.hmm.n_state
    kl_matrix = np.log10(Mouse.hmm.kl_divergence + 1)
    mouse_pos['state'] = pd.Series(Mouse.hmm.states, index = mouse_pos.index)
    
    Pellets = Mouse.arena.pellets.index
    Visits = Mouse.arena.visits.dropna(subset=['speed'])
    Starts = Visits['start']
    Ends = Visits['end']
    Entry = Mouse.arena.entry 

    colors = sns.xkcd_palette(color_names[0:N])
    cmap = gradient_cmap(colors)
    
    if pellet_delivery: Characterize_Timepoints('Pellet Delivery', Pellets, left_seconds = 3, right_seconds = 3, file_name = 'PelletDelivery')
    if start_visit: Characterize_Timepoints('Move Wheel', Starts, left_seconds = 5, right_seconds = 3, file_name = 'EnterVisit')
    if end_visit: Characterize_Timepoints('Leave Wheel', Ends, left_seconds = 3, right_seconds = 5, file_name = 'EndVisit')
    if enter_arena: Characterize_Timepoints('Enter Arena', Entry, left_seconds = 5, right_seconds = 5, file_name = 'EnterArena')
    
    print('Display_HMM_States_Characterization Completed')


'''-------------------------------PREDICTION-------------------------------'''  

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
            elif time_shift > 0:
                p[:-time_shift] = np.exp(-((characterized_state[time_shift:] - mean) ** 2) / (2 * variance ** 2))
            else:
                p = np.exp(-((characterized_state - mean) ** 2) / (2 * variance ** 2))
            p_.append(p)

        for p in p_: P *= p
        return P

    def Summarize(event_name, Events):
        mouse_pos_ = mouse_pos.copy()[Active_Chunk[0]:Active_Chunk[1]]
        Events = Events[Events > Active_Chunk[0]]
        Events = Events[Events < Active_Chunk[1]]

        COUNT_CURVES = [[] for _ in range(N)]
        COUNT_CURVES_MAX = []
        for i in range(N):
            count_curves = Mouse.hmm.process_states.Event_Triggering(mouse_pos_, Events, left_seconds = 5, right_seconds = 5, variable = 'State' + str(i), insert_nan = 0)
            COUNT_CURVES[i] = np.mean(np.array(count_curves), axis = 0)
            COUNT_CURVES_MAX.append(np.max(COUNT_CURVES[i]))

        Peaks_index = np.argsort(COUNT_CURVES_MAX)
        if COUNT_CURVES_MAX[Peaks_index[-4]] > 0.4: threshold = COUNT_CURVES_MAX[Peaks_index[-4]]
        elif COUNT_CURVES_MAX[Peaks_index[-3]] < 0.4: threshold = COUNT_CURVES_MAX[Peaks_index[-3]]
        else: threshold = 0.4
        
        characterized_states, characterized_states_peak, characterized_states_names, time_shifts = [], [], [], []
        for i in range(N):
            if np.max(COUNT_CURVES[i]) > threshold:
                characterized_states.append(i)
                characterized_states_peak.append(np.max(COUNT_CURVES[i]))
                characterized_states_names.append('State'+str(i))
                max_index = np.argsort(COUNT_CURVES[i], -1)[-1]
                time_shifts.append(max_index - 50 + 2)
        return characterized_states, characterized_states_peak, characterized_states_names, time_shifts
        
    def Predict(event_name, Events, file_name):
        characterized_states, characterized_states_peak, characterized_states_names, time_shifts = Summarize(event_name, Events)
        means = [1, 1,1,1,1,1, 1, 1,1,1,1,1]
        variances = [0.1, 0.1, 0.1,0.1,0.1, 0.1, 0.1, 0.1, 0.1,0.1,0.1, 0.1]
        
        if len(characterized_states) == 0:
            return 'No states characterized'

        characterized_states_curve = []
        for i in range(len(characterized_states)):
            characterized_states_curve.append(mouse_pos.loc[mouse_pos.index, characterized_states_names[i]].to_numpy())
        probability_curve = Calculate_Probability_Curve(characterized_states_curve, time_shifts = time_shifts, means = characterized_states_peak, variances = variances)
        mouse_pos.loc[mouse_pos.index, 'prob'] = probability_curve

        COUNT_CURVES = [[] for _ in range(len(characterized_states_curve))]
        Events = Events[Events > Predicting_Chunk_start]
        PROB = Mouse.hmm.process_states.Event_Triggering(mouse_pos, Events, left_seconds = 5, right_seconds = 5, variable = 'prob', insert_nan = 0)
        for i in range(len(characterized_states_curve)):
            COUNT_CURVES[i] = Mouse.hmm.process_states.Event_Triggering(mouse_pos, Events, left_seconds = 5, right_seconds = 5, variable = characterized_states_names[i], insert_nan = 0)
        
        fig, axs = plt.subplots(1, 1, figsize=(10, 6))
        T = np.arange(-50, 50, 1)
        for i in range(len(characterized_states_curve)):
            axs.plot(T, np.mean(np.array(COUNT_CURVES[i]), axis = 0), label = characterized_states_names[i], color = color_names[int(characterized_states_names[i][-1])], linestyle = '--')
        axs.axvline(x = 0, color = 'red')
        axs.legend(loc = 'upper right')
        axs_ = axs.twinx()
        axs_.plot(T, np.mean(np.array(PROB), axis = 0), color = 'black', label = 'Pred.')
        axs_.legend(loc = 'lower right')
        plt.savefig('../Images/Social_HMM/' + file_name + '/' + Mouse_title + '_Prediction.png')
        
        fig, axs = plt.subplots(1, 1, figsize=(50, 4))
        mouse_pos_ = mouse_pos[Predicting_Chunk_start:]
        mouse_pos_.prob.plot(ax = axs, color = 'black')
        for i in range(len(Events)):
            trigger = Events[i]
            axs.axvline(x = trigger, color = 'red')
        plt.savefig('../Images/Social_HMM/' + file_name + '/' + Mouse_title + '_Prediction_Full.png')
        
        return 'Predicton for ' + file_name + ' Completed'
    
    Mouse_title = Mouse.type + '_' + Mouse.mouse
    mouse_pos = Mouse.mouse_pos
    N = Mouse.hmm.n_state
    states = Mouse.hmm.states
    mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)

    Pellets = Mouse.arena.pellets.index
    Visits = Mouse.arena.visits
    Entry = Mouse.arena.entry 
    Starts = Visits['start'].to_numpy()
    Ends = Visits['end'].to_numpy()

    Active_Chunk = Mouse.active_chunk
    Predicting_Chunk_start = Active_Chunk[1] + pd.Timedelta('1S')
    
    mouse_pos = Mouse.hmm.process_states.State_Timewindow(mouse_pos, timewindow = 10)

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

def Display_HMM_States_Predicting_Behavior_Poisson(Mouse, pellet_delivery = True, start_visit = True, end_visit = True, enter_arena = True):
    
    Mouse_title = Mouse.type + '_' + Mouse.mouse
    mouse_pos = Mouse.mouse_pos
    N = Mouse.hmm.n_state
    states = Mouse.hmm.states
    mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)

    Pellets = Mouse.arena.pellets.index
    Entry = Mouse.arena.entry 
    Starts = Mouse.arena.visits['start'].to_numpy()
    Ends = Mouse.arena.visits['end'].to_numpy()

    Active_Chunk = [Mouse.active_chunk[0], Mouse.active_chunk[0] + pd.Timedelta('12H')] # 12 H, 7am-7pm
    Predicting_Chunk = [Mouse.active_chunk[1] + pd.Timedelta('1S') + pd.Timedelta('2H'), Mouse.active_chunk[1] + pd.Timedelta('1S') + pd.Timedelta('9H')]
    
    mouse_pos = Mouse.hmm.process_states.State_Timewindow(mouse_pos, timewindow = 100) # occupancy probability in the past 10 second
    
    for i in range(N):
        mouse_pos['State' + str(i) + '_2'] = mouse_pos['State' + str(i)].to_numpy() ** 2
    mouse_pos.loc[:, 'interc'] = np.ones(len(mouse_pos))
    
    def Shuffle(regression, prediction, Params):            
        Ls = []
        for i in range(1000):
            permutation = np.random.permutation(len(regression.X))
            regression.X = regression.X.iloc[permutation].set_index(regression.X.index)
            
            model = sm.GLM(regression.Y, regression.X, family=sm.families.Poisson(sm.families.links.Log()))
            #result = model.fit_regularized(alpha=0.01, L1_wt=1.0, maxiter = 1000)
            result = model.fit()
            y_pred = result.predict(prediction.X)
            y, y_pred = prediction.Y.to_numpy().reshape(1,-1)[0], y_pred.to_numpy()
                
            epsilon = 1e-10
            y_pred = np.where(y_pred <= 0, epsilon, y_pred)

            Ls.append(np.mean(y * np.log(y_pred) - y_pred - np.log(factorial(y))))
            
            for j in range(len(regression.regressor)):
                Params[j][i + 1] = result.params[regression.regressor[j]]
            
            print(i)
        return Ls, Params
    
    def Predict(event_name, Events, file_name):
        mouse_pos_ = mouse_pos.copy()[::100] # dt = 10s
        print(mouse_pos_.index[1] - mouse_pos_.index[0])
        
        mouse_pos_[file_name] = 0
        nearest_indices = mouse_pos_.index.get_indexer(Events, method='nearest')
        for i in range(len(nearest_indices)):
            mouse_pos_.iloc[nearest_indices[i], mouse_pos_.columns.get_loc(file_name)] += 1
        print('Locate events')
        
        time_period = 6 # 6*dt = 60s
        mouse_pos_['intensity'] = mouse_pos_[file_name].iloc[::-1].rolling(window=time_period, min_periods=1).sum().iloc[::-1].fillna(0)
        print('Cauculate intensity')
    
        mouse_pos_train = mouse_pos_[Active_Chunk[0]:Active_Chunk[1]]
        mouse_pos_test = mouse_pos_[Predicting_Chunk[0]:Predicting_Chunk[1]]
        
        regression = mouse.Regression(mouse_pos_train)
        regression.predictor = 'intensity'
        regression.regressor = ['State' + str(i) for i in range(N)] + ['State' + str(i) + '_2' for i in range(N)] + ['interc']
        result = regression.Poisson()
        
        print(result.summary())
        
        Params = np.zeros((len(regression.regressor), 1001))
        for i in range(len(regression.regressor)):
            Params[i][0] = result.params[regression.regressor[i]]
        #np.save('regression_params.npy', result.params)
        print('model training')
        
        prediction = mouse.Regression(mouse_pos_test)
        prediction.predictor = 'intensity'
        prediction.regressor = regression.regressor 
        prediction.Get_Variables()
        y_pred = result.predict(prediction.X)
        y, y_pred = prediction.Y.to_numpy().reshape(1,-1)[0], y_pred.to_numpy()
        print('prediction')
            
        epsilon = 1e-10
        y_pred = np.where(y_pred <= 0, epsilon, y_pred)
        L = np.mean(y * np.log(y_pred) - y_pred - np.log(factorial(y)))
        mouse_pos_test.loc[mouse_pos_test.index, 'intensity_pred'] = y_pred
        
        fig, axs = plt.subplots(1, 1, figsize=(50, 8), sharex=True)
        mouse_pos_test.intensity.plot(ax = axs, color = 'red', linewidth = 2.5, label = 'Observed')
        axs.set_ylabel('$\lambda(t)$', fontsize = 20)
        axs.legend(fontsize=20, loc = 'upper left')
        axs.tick_params(axis='both', which='major', labelsize=16)
        axs.spines['top'].set_visible(False)
        x_min, x_max = axs.get_xlim()
        axs.set_xlim(left=x_min, right=x_max)
        
        axs_ = axs.twinx()
        mouse_pos_test.intensity_pred.plot(ax = axs_, color = 'black', label = 'Predicted: Ave Log-Likelihood = ' + str(round(L, 3)))
        axs_.set_ylabel('$\hat{\lambda}(t)$', fontsize = 20)
        axs_.legend(fontsize=20, loc = 'upper right')
        axs_.tick_params(axis='both', which='major', labelsize=16)
        axs_.spines['top'].set_visible(False)
        axs_.set_xlim(left=x_min, right=x_max)
        
        plt.tight_layout()
        plt.savefig('../Images/Social_HMM_Prediction/' + file_name + '/' + Mouse_title + '.png')
        print('Plot Prediction')
        '''
        Ls, Params = Shuffle(regression, prediction, Params)
        print('Shuffle')
        
        t_stat, p_value = stats.ttest_1samp(Ls, L)
        fig, axs = plt.subplots(1, 1, figsize=(6, 8))
        axs.hist(Ls, bins = 20, color = 'blue', alpha = 0.8)
        axs.axvline(x=L, color = 'red', label = 'p-value: ' + str(round(p_value,3)))
        axs.legend(fontsize = 16, loc = 'upper left')
        axs.tick_params(axis='both', which='major', labelsize=12)
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig('../Images/Social_HMM_Prediction/' + file_name + '/' + Mouse_title + '_Validation.png')
        print('Plot Validation')
        
        for i in range(len(regression.regressor)):
            t_stat, p_value = stats.ttest_1samp(Params[i][1:], Params[i][0])
            print(regression.regressor[i], 'coefficient = ' + str(Params[i][0]), 'p value = ' + str(p_value))'''
        
        print('Predicton for ' + file_name + ' Completed') 
    
    if pellet_delivery: 
        Predict(event_name = 'Pellet Delivery', Events = Pellets, file_name = 'PelletDelivery')
    if start_visit: 
        Predict(event_name = 'Move Wheel', Events = Starts, file_name = 'EnterVisit')
    if end_visit: 
        Predict(event_name = 'Leave Wheel', Events = Ends, file_name = 'EndVisit')
    if enter_arena: 
        Predict(event_name = 'Enter Arena', Events = Entry, file_name = 'EnterArena')
    
    print('Display_HMM_States_Predicting_Behavior_Poisson Completed')

def Display_HMM_States_Predicting_Behavior_Logistic(Mouse, pellet_delivery = True, start_visit = True, end_visit = True, enter_arena = True):

    def Predict(event_name, Events, time_period, file_name):
        time_period = int(time_period/0.1) # dt = 0.1 s 
        mouse_pos_ = mouse_pos.copy()[::time_period]
        
        mouse_pos_['Event'] = 0
        nearest_indices = mouse_pos_.index.get_indexer(Events, method='nearest')
        for i in range(len(nearest_indices)):
            mouse_pos_.iloc[nearest_indices[i], mouse_pos_.columns.get_loc('Event')] = 1
        print('Locate events')
        
        '''
        mouse_pos['intensity'] = mouse_pos[file_name].iloc[::-1].rolling(window=time_period, min_periods=1).sum().iloc[::-1].fillna(0)
        mouse_pos['intensity'] = mouse_pos['intensity'].clip(upper=1)
        print('Cauculate intensity')
        '''
            
        mouse_pos_train = mouse_pos_[Active_Chunk[0]:Active_Chunk[1]]
        mouse_pos_test = mouse_pos_[Predicting_Chunk[0]:Predicting_Chunk[1]]
        
        regression = mouse.Regression(mouse_pos_train)
        regression.predictor = 'Event'
        regression.regressor = ['State' + str(i) for i in range(N)] + ['interc']
        result = regression.Logistic()
        print('Model training')
            
        prediction = mouse.Regression(mouse_pos_test)
        prediction.predictor = 'Event'
        prediction.regressor = regression.regressor 
        prediction.Get_Variables()
        y_pred = result.predict(prediction.X)
        accuracy = accuracy_score(mouse_pos_test['Event'], y_pred)
        mouse_pos_test.loc[:,'Pred'] = y_pred
        print('Model predicting')
        
        fig, axs = plt.subplots(2, 1, figsize=(50, 8))
        mouse_pos_test.Pred.plot(ax = axs[0], color = 'black', linewidth = 2.5, label = 'Predicted')
        mouse_pos_test.Event.plot(ax = axs[1], color = 'red', linewidth = 2.5, label = 'Observed')
        axs[0].plot([],[], color = 'white', label = 'Accuracy: ' + str(accuracy))
        axs[0].legend(fontsize=20, loc = 'upper left')
        for i in range(2):
            axs[i].set_ylabel('Event Occurence', fontsize = 20)
            axs[i].tick_params(axis='both', which='major', labelsize=16)
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            x_min, x_max = axs[i].get_xlim()
            axs[i].set_xlim(left=x_min, right=x_max)
        plt.tight_layout()
        plt.savefig('../Images/Social_HMM_Prediction/' + file_name + '/' + Mouse_title + '_Logist.png')
        print('Predicton for ' + file_name + ' Completed')
        
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

    Active_Chunk = [Mouse.active_chunk[0], Mouse.active_chunk[0] + pd.Timedelta('12H')] # 12 H, 7am-7pm
    Predicting_Chunk = [Mouse.active_chunk[1] + pd.Timedelta('1S') + pd.Timedelta('2H'), Mouse.active_chunk[1] + pd.Timedelta('1S') + pd.Timedelta('9H')]
    
    mouse_pos = Mouse.hmm.process_states.State_Timewindow(mouse_pos, timewindow = 20) # occupancy probability in the past 2 second
    mouse_pos.loc[:, 'interc'] = np.ones(len(mouse_pos))
    
    print('Start predicting')
    
    if pellet_delivery: 
        Predict(event_name = 'Pellet Delivery', Events = Pellets, time_period = 0.5, file_name = 'PelletDelivery')
    if start_visit: 
        Predict(event_name = 'Move Wheel', Events = Starts, time_period = 10, file_name = 'EnterVisit')
    if end_visit: 
        Predict(event_name = 'Leave Wheel', Events = Ends, time_period = 9, file_name = 'EndVisit')
    if enter_arena: 
        Predict(event_name = 'Enter Arena', Events = Entry, time_period = 2, file_name = 'EnterArena')
    
    print('Display_HMM_States_Predicting_Behavior_MLP Completed')

def Display_HMM_States_Predicting_Behavior_SeqDetect(Mouse, pellet_delivery = True, start_visit = True, end_visit = True, enter_arena = True):
    
    Mouse_title = Mouse.type + '_' + Mouse.mouse
    mouse_pos = Mouse.mouse_pos
    N = Mouse.hmm.n_state
    states = Mouse.hmm.states
    mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)

    Pellets = Mouse.arena.pellets.index
    Entry = Mouse.arena.entry 
    Starts = Mouse.arena.visits['start'].to_numpy()
    Ends = Mouse.arena.visits['end'].to_numpy()

    Active_Chunk = [Mouse.active_chunk[0], Mouse.active_chunk[0] + pd.Timedelta('12H')] # 12 H, 7am-7pm
    Predicting_Chunk = [Mouse.active_chunk[1] + pd.Timedelta('1S') + pd.Timedelta('2H'), Mouse.active_chunk[1] + pd.Timedelta('1S') + pd.Timedelta('9H')]

    def CompareEvents(STATES, sequence):
        ERROR = []
        dominant_sequence = sequence + 1
        for i in range(len(STATES)):
            state = STATES[i]+ 1
            #ERROR.append(np.mean(abs(state - dominant_sequence)))
            ERROR.append(dtw.distance(state, dominant_sequence))
        mu, sigma = np.mean(ERROR), np.std(ERROR)
        return(mu - 1*sigma, mu+1*sigma)

    def Predict(event_name, Events, left_seconds, file_name):
        STATES = Mouse.hmm.process_states.Event_Triggering(mouse_pos, Events, left_seconds, 0, 'state', insert_nan = 0)
        sequence = Mouse.hmm.process_states.Find_Event_Sequence(STATES)
        
        trigger = int(left_seconds*10)
        params = CompareEvents(STATES, sequence)
        lower_bound, upper_bound = params[0], params[1]
        print('Calculate errors')
        
        mouse_pos['Event'] = 0
        nearest_indices = mouse_pos.index.get_indexer(Events, method='nearest')
        for i in range(len(nearest_indices)):
            mouse_pos.iloc[nearest_indices[i], mouse_pos.columns.get_loc('Event')] = 1
        print('Locate events')
        
        mouse_pos_test = mouse_pos[Predicting_Chunk[0]:Predicting_Chunk[1]]
        states = mouse_pos_test['state'].to_numpy()
        pred = np.zeros(len(mouse_pos_test))
        for i in range(len(states)-trigger):
            state = states[i:i+trigger] + 1
            #error = np.mean(abs(state - (sequence_[0] + 1)))
            error = dtw.distance(state, (sequence + 1))
            if error > lower_bound and error < upper_bound: pred[i] = 1
        mouse_pos_test['Pred'] = pred
        print('Model predicting')
        
        fig, axs = plt.subplots(2, 1, figsize=(50, 8))
        mouse_pos_test.Pred.plot(ax = axs[0], color = 'black', linewidth = 2.5, label = 'Predicted')
        mouse_pos_test.Event.plot(ax = axs[1], color = 'red', linewidth = 2.5, label = 'Observed')
        axs[0].legend(fontsize=20, loc = 'upper left')
        for i in range(2):
            axs[i].set_ylabel('Event Occurence', fontsize = 20)
            axs[i].tick_params(axis='both', which='major', labelsize=16)
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            x_min, x_max = axs[i].get_xlim()
            axs[i].set_xlim(left=x_min, right=x_max)
        plt.tight_layout()
        plt.savefig('../Images/Social_HMM_Prediction/' + file_name + '/' + Mouse_title + '_SeqDetect.png')
        print('Predicton for ' + file_name + ' Completed')

    if pellet_delivery: 
        Predict(event_name = 'Pellet Delivery', Events = Pellets, left_seconds = 1, file_name = 'PelletDelivery')
    if start_visit: 
        Predict(event_name = 'Move Wheel', Events = Starts, left_seconds = 2, file_name = 'EnterVisit')
    if end_visit: 
        Predict(event_name = 'Leave Wheel', Events = Ends, left_seconds = 2, file_name = 'EndVisit')
    if enter_arena: 
        Predict(event_name = 'Enter Arena', Events = Entry, left_seconds = 1, file_name = 'EnterArena')
    
    print('Display_HMM_States_Predicting_Behavior_SeqDetect Completed')

'''-------------------------------REGRESSION-------------------------------'''  
def Display_Visit_Prediction(VISITS, model, file_path, title):
    regression = mouse.Regression(VISITS)
    regression.regressor = ['speed', 'acceleration', 'last_pellets_self', 'last_pellets_other','last_duration', 'last_interval','last_pellets_interval', 'entry']
    
    if model == 'linear': 
        obs, pred, result = regression.Linear_Regression()
        fig, axs = plt.subplots(figsize=(20, 8))
        axs.axis('off')
        axs.text(0.5, 0.5, str(result.summary()),
                    verticalalignment='center', horizontalalignment='left',
                    transform=axs.transAxes, fontsize=12)
        plt.savefig(file_path + 'Model_' + title)
    if model == 'MLP': 
        obs, pred = regression.Multilayer_Perceptron()
    
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    axs.scatter(obs, pred, s = 10)    
    axs.set_xlabel('Observation', fontsize = 24)
    axs.set_ylabel('Prediction', fontsize = 24)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    #axs.legend(fontsize = 20)
    #axs.set_ylim((-max(obs)-1, max(obs)+1))
    plt.tight_layout()
    plt.savefig(file_path + title)

def main():
    VISITS = []
    for label in LABELS:
        aeon_exp_name, type_name, mouse_name = label[0], label[1], label[2]
        print('Start Processing: ', type_name, "-", mouse_name)
        
        Mouse = mouse.Mouse(aeon_exp = aeon_exp_name, type = type_name, mouse = mouse_name)
        Mouse.Run_Visits()
        
        '''-------------------------------BODY-------------------------------'''
        
        NODES = [['head', 'spine3'],['spine1', 'spine3'],['left_ear', 'spine3'],['right_ear', 'spine3']]
        for nodes in NODES:
            Mouse.Add_Body_Info_to_mouse_pos(property = 'distance', nodes = nodes)
            #Display_Body_Info(Mouse, property = 'distance', nodes = nodes)
        '''
        Display_Body_Info_Characterization(Mouse, NODES,
                                            pellet_delivery = True,
                                            start_visit = True,
                                            end_visit = True,
                                            enter_arena = True)'''

    
        '''-------------------------------LDS-------------------------------'''
        
        '''
        
        Display_LDS_Trace(Mouse, file_path = '../Images/Social_LDS/')
        Display_Kinematics_Distribution_Along_Time(Mouse, file_path = '../Images/Social_LDS/Distribution_')
        Display_Kinematics_Properties_Along_Time(Mouse,  file_path = '../Images/Social_LDS/Properties_')'''
        '''
        '''
        '''-------------------------------HMM-------------------------------'''
        #Display_Model_Selection(Mouse, N = np.arange(3, 27), file_path = '../Images/Social_HMM/StateNumber/')
        
        
        #Mouse.hmm.Fit_Model_without_Saving(n_state = 20, feature = 'Kinematics_and_Body')
        
        #Mouse.hmm.Fit_Model(n_state = 10, feature = 'Kinematics_and_Body')
        Mouse.hmm.Get_States(n_state = 10, feature = 'Kinematics_and_Body')
        
        
        Display_HMM_TransM(Mouse, file_path = '../Images/Social_HMM/TransM/', exclude_diag = True)
        #Display_HMM_KLMatrix(Mouse, file_path = '../Images/Social_HMM/KL_Matrix/')
        '''Display_HMM_States_Along_Time(Mouse, file_path = '../Images/Social_HMM/State/') 
        Display_HMM_States_Duration_Along_Time(Mouse, file_path = '../Images/Social_HMM/State/') 
        
        Display_HMM_States_Feature(Mouse, file_path = '../Images/Social_HMM/')
        
        
        Display_HMM_States_Characterization(Mouse, 
                                            pellet_delivery = True,
                                            start_visit = True,
                                            end_visit = True,
                                            enter_arena = True,
                                            file_path = '../Images/Social_HMM/')
        '''
        '''
        Display_HMM_States_Predicting_Behavior_Gaussian(Mouse,
                                                        pellet_delivery = True,
                                                        start_visit = True,
                                                        end_visit = True,
                                                        enter_arena = True)
        
        
        Display_HMM_States_Predicting_Behavior_Logistic(Mouse,
                                                    pellet_delivery = True,
                                                    start_visit = True,
                                                    end_visit = True,
                                                    enter_arena = True)
        
        
        
        Display_HMM_States_Predicting_Behavior_Poisson(Mouse,
                                                        pellet_delivery = True,
                                                        start_visit = True,
                                                        end_visit = False,
                                                        enter_arena = False)
        
        
        Display_HMM_States_Predicting_Behavior_SeqDetect(Mouse,
                                                    pellet_delivery = True,
                                                    start_visit = True,
                                                    end_visit = True,
                                                    enter_arena = True)
        '''
        '''-------------------------------REGRESSION-------------------------------'''                                          

        '''Display_Visit_Prediction(Mouse.arena.visits, model = 'linear', file_path = '../Images/Social_Regression/'+Mouse.type+'-'+Mouse.mouse+'/', title = 'Linear_Regression.png')                                            
        Display_Visit_Prediction(Mouse.arena.visits, model = 'MLP', file_path = '../Images/Social_Regression/'+Mouse.type+'-'+Mouse.mouse+'/', title = 'MLP.png')
        
        
        VISITS.append(Mouse.arena.visits)
        
    VISITS = pd.concat(VISITS, ignore_index=True)
    Display_Visit_Prediction(VISITS, model = 'linear', file_path = '../Images/Social_Regression/All-Mice/', title = 'Linear_Regression.png')                                            
    Display_Visit_Prediction(VISITS, model = 'MLP', file_path = '../Images/Social_Regression/All-Mice/', title = 'MLP.png')'''

if __name__ == "__main__":
        main()
        
        
        
        