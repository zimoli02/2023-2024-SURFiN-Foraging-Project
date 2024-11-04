import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import pandas as pd
import seaborn as sns
import pickle

import statsmodels.api as sm
import scipy.stats as stats
import random
from scipy.special import factorial
from dtaidistance import dtw
from scipy.linalg import inv, det

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from collections import Counter

import sys
from pathlib import Path
current_script_path = Path(__file__).resolve()

functions_dir = current_script_path.parents[1] / 'Functions'
sys.path.insert(0, str(functions_dir))
import mouse as mouse
import result as result

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

nodes_name = ['nose', 'head', 'right_ear', 'left_ear', 'spine1', 'spine2','spine3', 'spine4']
color_names = [
    'black', 'blue', 'red', 'tan', 'green', 'brown', 
    'purple', 'orange', 'magenta', 'olive', 'pink', 
    'darkblue', 'lime', 'cyan', 'turquoise', 'gold', 
    'navy', 'maroon', 'teal', 'grey']

Mice_N = np.array([0,1,3,4,5,6,7])

def kl_divergence_gaussian(p_means, p_variances, q_means, q_variances):
    k = len(p_means)

    q_variances_inv = inv(q_variances)
    trace_term = np.trace(np.dot(q_variances_inv, p_variances))
    
    mean_diff = q_means - p_means
    mean_term = np.dot(np.dot(mean_diff.T, q_variances_inv), mean_diff)

    det_term = np.log(det(q_variances) / det(p_variances))

    kl_div = max(0, 0.5 * (trace_term + mean_term - k + det_term))
    
    return kl_div

def create_kl_divergence_matrix(states):
    n_states = len(states)
    kl_matrix = np.zeros((n_states, n_states))
    
    for i in range(n_states):
        for j in range(n_states): 
            p_means, p_variances = states[i]
            q_means, q_variances = states[j]
            kl_matrix[i, j] = kl_divergence_gaussian(p_means, p_variances, q_means, q_variances)

    return kl_matrix

def Compare_Sequence(seq1, seq2, kl_matrix):
    nx, ny = len(seq1), len(seq2)

    cost = np.zeros((nx + 1, ny + 1))
    cost[0, 1:] = np.inf
    cost[1:, 0] = np.inf

    for i in range(1, nx + 1):
        j_start, j_stop = 1, ny + 1
        for j in range(j_start, j_stop):
            cost[i, j] = kl_matrix[int(seq1[i-1]), int(seq2[j-1])]
            cost[i, j] += min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
    
    return cost[-1, -1]

def Exclude_Diagonal(matrix):
    return matrix[~np.eye(matrix.shape[0], dtype=bool)].reshape(matrix.shape[0], -1).reshape(-1)


def Combine_Mice():
    Mice = []
    NODES = [['head', 'spine3'],['spine1', 'spine3'],['left_ear', 'spine3'],['right_ear', 'spine3']]
    for label in LABELS:
        aeon_exp_name, type_name, mouse_name = label[0], label[1], label[2]
        print('Start Processing: ', type_name, "-", mouse_name)
        
        Mouse = mouse.Mouse(aeon_exp = aeon_exp_name, type = type_name, mouse = mouse_name)
        Mouse.Run_Visits()
        
        Mouse.hmm.Get_States(n_state = 10, feature = 'Kinematics_and_Body')
        Mouse.mouse_pos['state'] = Mouse.hmm.states
    
        Mice.append(Mouse)
    
    with open('../Report/Data/Mice.pkl', 'wb') as file:
        pickle.dump(Mice, file)
    
    print('Complete Saving')
    
    return Mice
'''-------------------------------Kinematics-------------------------------'''
def Display_LDS_Parameters(Mice):
    data = []
    for i in range(len(Mice_N)):
        Mouse = Mice[Mice_N[i]]
        exp_session = mouse.Session(aeon_exp = Mouse.aeon_exp, type = Mouse.type, mouse = Mouse.mouse, start = Mouse.starts[0], end = Mouse.ends[0])
        exp_session.kinematics.Run(Mouse)
        mouse_lds_parameters = exp_session.kinematics.parameters
        sigma_a = mouse_lds_parameters['sigma_a']
        sigma_x = mouse_lds_parameters['sigma_x']
        sigma_y = mouse_lds_parameters['sigma_y']
        
        data.append({
            'aeon_exp_name': Mouse.aeon_exp,
            'type_name': Mouse.type,
            'mouse_name': Mouse.mouse,
            'sigma_a': abs(sigma_a),
            'sigma_x': abs(sigma_x),
            'sigma_y': abs(sigma_y)
        })
    df = pd.DataFrame(data)
    return df
    
def Display_Kinematics_Properties_Along_Time(Mice, file_path):   
    
    Full_Mean_V, Full_Variance_V, Full_Mean_A, Full_Variance_A = np.full((len(Mice_N),24*4), np.nan),np.full((len(Mice_N),24*4), np.nan),np.full((len(Mice_N),24*4), np.nan),np.full((len(Mice_N),24*4), np.nan)
    PHASE = []
    for i in range(len(Mice_N)):
        Mouse = Mice[Mice_N[i]]
        mouse_pos = Mouse.mouse_pos
        PHASE.append(Mouse.type)
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

        day = 0
        for j in range(len(starts)):
            hour = starts[j].hour
            if hour == 0: day += 1
            
            index_in_full_sequence = day*24 + hour
            
            df = mouse_pos[starts[j]:ends[j]]
            if len(df) == 0: continue
            
            speed = df.smoothed_speed
            Full_Mean_V[i][index_in_full_sequence] = np.mean(speed)
            Full_Variance_V[i][index_in_full_sequence] = np.var(speed)
                
            acce = df.smoothed_acceleration
            Full_Mean_A[i][index_in_full_sequence] = np.mean(acce)
            Full_Variance_A[i][index_in_full_sequence] = np.var(speed)

        np.save('../Report/Data/Full_Mean_V.npy', Full_Mean_V)
        np.save('../Report/Data/Full_Variance_V.npy', Full_Variance_V)
        np.save('../Report/Data/Full_Mean_A.npy', Full_Mean_A)
        np.save('../Report/Data/Full_Variance_A.npy', Full_Variance_A)

    Full_Hour = np.array([i % 24 for i in range(24*4)])
    Full_Sequence = np.arange(4*24)
    Full_CR = np.array([[7 + 24*i, 19 + 24*i] for i in range(4)])

    fig, axs = plt.subplots(2, 1, figsize = (12, 16))
    for i in range(len(PHASE)):
        if PHASE[i] == 'Pre': colors = 'black'
        else: colors = 'red'
        #colors = colors_name[i]
        axs[0].plot(Full_Sequence, Full_Mean_V[i], color = colors)
        axs[1].plot(Full_Sequence, Full_Variance_V[i], color = colors)
    #axs[0].set_ylim(0,75)
    #axs[1].set_ylim(0,12000)
    axs[0].set_ylabel('Mean', fontsize = 40)
    axs[1].set_ylabel('Variance', fontsize = 40)
    axs[1].set_xlabel('Hours', fontsize = 40)
    for i in range(2):
        axs[i].plot([], [], color = 'black', label = 'Pre-Social')
        axs[i].plot([], [], color = 'red', label = 'Post-Social')
        axs[i].legend(loc = 'upper right', fontsize = 25)
        axs[i].set_xticks(Full_Sequence[::6], Full_Hour[::6])
        axs[i].tick_params(axis='both', which='major', labelsize=25)
        for t in range(len(Full_CR)):
            axs[i].axvspan(Full_CR[t][0],Full_CR[t][1], color='lightblue', alpha=0.5)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
    axs[0].set_title('Speed', fontsize = 45)
    plt.tight_layout()
    plt.savefig(file_path + 'Speed.png')

    fig, axs = plt.subplots(2, 1, figsize = (12, 16))
    for i in range(len(PHASE)):
        if PHASE[i] == 'Pre': colors = 'black'
        else: colors = 'red'
        #colors = colors_name[i]
        axs[0].plot(Full_Sequence, Full_Mean_A[i], color = colors)
        axs[1].plot(Full_Sequence, Full_Variance_A[i], color = colors)
    #axs[0].set_ylim(0,150)
    #axs[1].set_ylim(0,100000)
    axs[0].set_ylabel('Mean', fontsize = 40)
    axs[1].set_ylabel('Variance', fontsize = 40)
    axs[1].set_xlabel('Hours', fontsize = 40)
    for i in range(2):
        axs[i].plot([], [], color = 'black', label = 'Pre-Social')
        axs[i].plot([], [], color = 'red', label = 'Post-Social')
        axs[i].legend(loc = 'upper right', fontsize = 25)
        axs[i].set_xticks(Full_Sequence[::6], Full_Hour[::6])
        axs[i].tick_params(axis='both', which='major', labelsize=25)
        for t in range(len(Full_CR)):
            axs[i].axvspan(Full_CR[t][0],Full_CR[t][1], color='lightblue', alpha=0.5)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
    axs[0].set_title('Acceleration', fontsize = 45)
    plt.tight_layout()
    plt.savefig(file_path + 'Acce.png')

    print('Display_Kinematics_Properties_Along_Time Completed')


def Display_Kinematics_Comparison(Mice, file_path):
    speed, acce = [], []
    mice = []
    for i in range(len(Mice_N)):
        Mouse = Mice[Mice_N[i]]
        
        start, end = Mouse.active_chunk[0],  Mouse.active_chunk[1]
        df = Mouse.mouse_pos[start:end]
        speed.append(df.smoothed_speed)
        acce.append(df.smoothed_acceleration)
        mice.append(Mouse.type + '-' + Mouse.name[-2:])

    diff = np.zeros((len(mice), len(mice)))
    for i in range(len(mice)):
        for j in range(len(mice)):
            data_A = speed[i]
            data_B = speed[j]

            ks_stat, ks_p_value = stats.ks_2samp(data_A, data_B)
            diff[i][j] = ks_p_value

    fig, axs = plt.subplots(1,1, figsize=(10,8))
    sns.heatmap(diff, cmap='RdBu', ax = axs, square = 'True', cbar = True)
    axs.set_title("Comparison between Mice Speed", fontsize = 25)
    axs.set_xticklabels(mice)
    axs.set_yticklabels(mice, rotation = 0)
    axs.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig(file_path + 'Compare_Speed.png')
    
    fig, axs = plt.subplots(1,1, figsize=(10,8))
    for i in range(len(mice)):
        data = speed[i]
        if LABELS[i][1] == 'Pre': color = 'black'
        else: color = 'red'
        
        axs.scatter(np.mean(data), np.var(data), color = color)
    axs.scatter([],[],color = 'black', label = 'Pre')
    axs.scatter([],[],color = 'red', label = 'Post')
    axs.legend()
    axs.set_xticklabels(mice)
    axs.set_yticklabels(mice, rotation = 0)
    axs.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig(file_path +'Compare_Speed_and_Acce.png')
    
    print('Complete Display_Kinematics_Comparison')


'''-------------------------------Behaviour States-------------------------------'''   
def Display_Model_Selection(Mice, N, file_path):
    points = 24*60*60*10
    Ls = []
    for i in range(len(Mice_N)):
        Mouse = Mice[Mice_N[i]]
        Mouse_title = Mouse.type + '_' + Mouse.name
        Loglikelihood = np.load('../../SocialData/HMMStates/Loglikelihood_' + Mouse_title + '.npy', allow_pickle=True)
        Ls.append(Loglikelihood/points)
    
    N = np.arange(3, 27)
    fig, axs = plt.subplots(1,1,figsize = (10,8))
    for i in range(len(Ls)):
        Loglikelihood = Ls[i]
        df = Loglikelihood[1:] - Loglikelihood[:-1]
        axs.scatter(N[1:], df, color = color_names[i])
        axs.plot(N[1:], df, color = color_names[i])
    axs.axvline(x=10, color = 'red', linestyle = "--")
    axs.set_xticks(N[1:])
    axs.set_ylabel('$\Delta$Log Likelihood per Point', fontsize = 20)
    axs.set_xlabel('State Number', fontsize = 20)
    axs.tick_params(axis='both', which='major', labelsize=12)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(file_path + 'StateNumberSelection.png')
    
    print('Complete Display_Model_Selection')

def Display_HMM_Similar_States_Comparison(Summary, Wheel, Grooming, file_path):

    K_L = True 
    Images = True 
    Parameter = True 
    
    if Wheel: 
        SingleState = Summary.Single_State(Summary, state_meaning = 'WheelMoving', states = [4,6,6,4,4,4,4,6])
        SingleState.Display(K_L, Images, Parameter, file_path)

    if Grooming: 
        SingleState = Summary.Single_State(Summary, state_meaning = 'Grooming', states = [5,4,4,5,6,3,3,3])
        SingleState.Display(K_L, Images, Parameter, file_path)
    
    print('Display_HMM_Similar_States_Comparison Completed')

def Display_HMM_States_Duration_Along_Time(Mice, file_path):
    StateFreq = np.zeros((10, len(Mice_N)))
    
    for i in range(len(Mice_N)):
        Mouse = Mice[Mice_N[i]]
        states = Mouse.hmm.states
        
        count = np.zeros(10)
        for num in states:
            count[num] += 1
        count = count/len(states)
        
        for j in range(10):
            StateFreq[j][i] = count[j]

    row_labels = ['S '+str(i+1) for i in range(Mouse.hmm.n_state)]
    column_labels = [str(i+1) for i in Mice_N ]
    fig, axs = plt.subplots(1,1,figsize = (20, 12))
    sns.heatmap(StateFreq, cmap='RdBu', ax = axs, square = 'True', cbar = False)
    axs.set_title("State Frequency", fontsize = 25)
    axs.set_xlabel('Mouse', fontsize = 16)
    axs.set_ylabel('State', fontsize = 16)
    axs.set_xticks(np.arange(0.5, len(column_labels)+0.5, 1))
    axs.set_xticklabels(column_labels, rotation = 0)
    axs.set_yticks(np.arange(0.5,len(row_labels)+0.5,1))
    axs.set_yticklabels(row_labels, rotation = 0)
    axs.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(file_path + 'StateFrequency.png')
    
    print('Complete Display_HMM_States_Duration_Along_Time')

def Display_Escape_Behaviour_Summary(Mice, file_path):
    Duration_Mean = []
    Duration_Var = []
    ESCAPES = []
    ESCAPES_Mice = []
    
    for i in range(len(Mice_N)):
        Mouse = Mice[Mice_N[i]]
        Visits = Mouse.arena.visits.dropna(subset=['speed'])
        Ends = Visits['end']
        
        left_seconds, right_seconds = 3, 5
        STATES = Mouse.hmm.process_states.Event_Triggering(Mouse.mouse_pos, Ends, left_seconds, right_seconds, 'state', insert_nan = 0)
        
        Escapes = []
        for state in STATES:
            escape_state = state[left_seconds*10]
            finish_escape = left_seconds*10
            for i in range(int(left_seconds*10), len(state)-1):
                if state[i] == escape_state and state[i+1]!=escape_state: 
                    finish_escape = i + 1
                    break
            Escapes.append((finish_escape - left_seconds*10)/10)
            ESCAPES.append((finish_escape - left_seconds*10)/10)
        ESCAPES_Mice.append(np.array(Escapes))
        
        Duration_Mean.append(np.mean(Escapes))
        Duration_Var.append(np.std(Escapes))
    np.save('../Report/Data/ESCAPES_Mice.npy', np.array(ESCAPES_Mice))
    
    Duration_Mean = np.array(Duration_Mean)[[0,1,2,3,5,4,6]]
    Duration_Var = np.array(Duration_Var)[[0,1,2,3,5,4,6]]
    Session_label = [Mice[i].mouse[-3:]+'_' + Mice[i].type for i in Mice_N[[0,1,2,3,5,4,6]]]
    
    fig, axs = plt.subplots(1,1,figsize = (10, 6))
    axs.bar(range(len(Duration_Mean)), Duration_Mean, yerr=Duration_Var, capsize=14)
    axs.set_ylabel('Duration (s)', fontsize = 30)
    axs.set_xticks(range(len(Duration_Mean)))
    axs.set_xticklabels(Session_label)
    axs.tick_params(axis='both', which='major', labelsize=16)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(file_path + 'EndVisitEscapeDuration_All.png')
    
    print(np.mean(ESCAPES))
    print(np.std(ESCAPES))
    
    print('Complete Display_Escape_Behaviour_Summary')
    
def Display_Distance_Comparison_Pre_Post(file_path = '../Images/AllMice/'):
    fig, axs = plt.subplots(1, 4, figsize=(32, 6))
    axs = axs.flatten()
    events = np.array(['PelletDelivery','EnterVisit', 'EndVisit', 'EnterArena'])
    mice = np.array(['BAA-1104047', 'BAA-1104048','BAA-1104049'])

    for i in range(len(events)):
        event = events[i]
        pre = np.array([])
        post = np.array([])
        for mouse in mice:
            pre_corre = np.load('../../SocialData/HMMStates/Corre_'+ event + '_Pre_' + mouse + '.npy', allow_pickle=True)
            post_corre = np.load('../../SocialData/HMMStates/Corre_'+ event + '_Post_' + mouse + ".npy", allow_pickle=True)
            pre = np.concatenate((pre,pre_corre))
            post = np.concatenate((post,post_corre))
        
        statistic, p_value = stats.ks_2samp(pre, post)
        print(event)
        print('Pre', np.mean(pre))
        print('Post', np.mean(post))
        axs[i].hist(pre, bins = 50, density = True, color = 'purple', alpha = 0.5, label = 'Pre')
        axs[i].hist(post,bins = 50, density = True, color = 'green', alpha = 0.5, label = 'Post')
        axs[i].scatter([],[], color = 'white', label = 'p value = ' + str(round(p_value, 3)))
        
        axs[i].legend(loc = 'upper right', fontsize = 20)
        
        axs[i].set_title(event, fontsize=24)
        #axs[i].set_xticks(np.arange(-1, 1.1, 0.5))
        axs[i].set_xlabel('Distance', fontsize=20)
        axs[i].set_ylabel('Count.', fontsize=20)
        #axs[i].set_ylim(0, 1.0)  # Adjust if needed
        
        # Increase tick label font size
        axs[i].tick_params(axis='both', which='major', labelsize=18)
        
        # Remove top and right spines
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(file_path + 'Compare_Pre_Post.png')

    print('Complete Display_Distance_Comparison')
    
def Display_Dominant_Sequence_Comparison_Across_Sessions(Summary, pellet_delivery, start_visit, end_visit, enter_arena, file_path):

    if pellet_delivery: 
        EventSequence = Summary.Event_Sequence(event_name = 'Pellet Delivery', left_seconds = 3, right_seconds = 3)
        EventSequence.Display(file_path, file_name = 'PelletDelivery')
    if start_visit: 
        EventSequence = Summary.Event_Sequence(event_name = 'Start Visit', left_seconds = 5, right_seconds = 3)
        EventSequence.Display(file_path, file_name = 'StartVisit')
    if end_visit: 
        EventSequence = Summary.Event_Sequence(event_name = 'End Visit', left_seconds = 3, right_seconds = 5)
        EventSequence.Display(file_path, file_name = 'EndVisit')
    if enter_arena: 
        EventSequence = Summary.Event_Sequence(event_name = 'EnterArena', left_seconds = 3, right_seconds = 3)
        EventSequence.Display(file_path, file_name = 'EnterArena')


'''-------------------------------REGRESSION-------------------------------'''  
def Display_Visit_Prediction(Mice, model, file_path, title):
    VISITS = []
    for i in range(len(Mice_N)):
        Mouse = Mice[Mice_N[i]]
        VISITS.append(Mouse.arena.visits)
    VISITS = pd.concat(VISITS, ignore_index=True)
    
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
    
    try:
        print('Start Loading')
        with open('../Report/Data/Mice.pkl', 'rb') as file:
            Mice = pickle.load(file)
        print('Complete Loading')
    except FileNotFoundError:
        print('Start Combine_Mice')
        Mice = Combine_Mice()

    Report = False

    if Report == True: file_path = '../Report/Download/'
    else: file_path = '../Images/AllMice/'
    
    Summary = result.Summary(Mice)
    
    '''-------------------------------Kinematics-------------------------------'''

    #Display_LDS_Parameters(Mice)

    #Display_Kinematics_Properties_Along_Time(Mice, file_path = file_path)
    
    #Display_Kinematics_Comparison(Mice, file_path = file_path)
    
    
    '''-------------------------------Behaviour-------------------------------'''
    #Display_Model_Selection(Mice, N = np.arange(3, 27), file_path = file_path)

    #Display_HMM_States_Duration_Along_Time(Mice, file_path = file_path) 
    
    Display_HMM_Similar_States_Comparison(Summary, Wheel = True, Grooming = True, file_path = file_path + 'SingleState/')
    
    #Display_Escape_Behaviour_Summary(Mice, file_path = file_path)
    
    #Display_Distance_Comparison_Pre_Post(file_path = file_path)
    
    Display_Dominant_Sequence_Comparison_Across_Sessions(Summary, pellet_delivery = True, start_visit = True, end_visit = True, enter_arena = True, file_path = file_path + 'EventSequence/')
    
        
    '''-------------------------------Regression-------------------------------'''
    #Display_Visit_Prediction(Mice, model = 'linear', file_path = '../Images/Social_Regression/All-Mice/', title = 'Linear_Regression.png')                                            
    #Display_Visit_Prediction(Mice, model = 'MLP', file_path = '../Images/Social_Regression/All-Mice/', title = 'MLP.png')

if __name__ == "__main__":
        main()
        
        
        
        