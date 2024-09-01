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

    kl_div = 0.5 * (trace_term + mean_term - k + det_term)
    
    return kl_div

def symmetric_kl_divergence_gaussian(p_means, p_variances, q_means, q_variances):
    kl_pq = kl_divergence_gaussian(p_means, p_variances, q_means, q_variances)
    kl_qp = kl_divergence_gaussian(q_means, q_variances, p_means, p_variances)
    return 0.5 * (kl_pq + kl_qp)

def create_kl_divergence_matrix(states):
    n_states = len(states)
    kl_matrix = np.zeros((n_states, n_states))
    
    for i in range(n_states):
        for j in range(i, n_states):  # We only need to compute half the matrix due to symmetry
            p_means, p_variances = states[i]
            q_means, q_variances = states[j]
            kl_div = symmetric_kl_divergence_gaussian(p_means, p_variances, q_means, q_variances)
            kl_matrix[i, j] = kl_div
            kl_matrix[j, i] = kl_div  # Symmetric, so we fill both sides
    
    return kl_matrix

def Combine_Mice():
    Mice = []
    NODES = [['head', 'spine3'],['spine1', 'spine3'],['left_ear', 'spine3'],['right_ear', 'spine3']]
    for label in LABELS:
        aeon_exp_name, type_name, mouse_name = label[0], label[1], label[2]
        print('Start Processing: ', type_name, "-", mouse_name)
        
        Mouse = mouse.Mouse(aeon_exp = aeon_exp_name, type = type_name, mouse = mouse_name)
        Mouse.Run_Visits()
        
        for nodes in NODES:
            Mouse.Add_Body_Info_to_mouse_pos(property = 'distance', nodes = nodes)
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
            'mouse_name': Mouse.name,
            'sigma_a': sigma_a,
            'sigma_x': sigma_x,
            'sigma_y': sigma_y
        })
    df = pd.DataFrame(data)
    return df
    
    
def Display_Kinematics_Properties_Along_Time(Mice, file_path):   
    try:
        Full_Mean_V  =  np.load('Data/Full_Mean_V.npy', allow_pickle=True)
        Full_Variance_V  =  np.load('Data/Full_Variance_V.npy', allow_pickle=True)
        Full_Mean_A  =  np.load('Data/Full_Mean_A.npy', allow_pickle=True)
        Full_Variance_A  =  np.load('Data/Full_Variance_A.npy', allow_pickle=True)
    except FileNotFoundError:
        Full_Mean_V, Full_Variance_V, Full_Mean_A, Full_Variance_A = np.full((len(Mice_N),24*4), np.nan),np.full((len(Mice_N),24*4), np.nan),np.full((len(Mice_N),24*4), np.nan),np.full((len(Mice_N),24*4), np.nan)

        for i in range(len(Mice_N)):
            Mouse = Mice[Mice_N[i]]
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

            np.save('Data/Full_Mean_V.npy', Full_Mean_V)
            np.save('Data/Full_Variance_V.npy', Full_Variance_V)
            np.save('Data/Full_Mean_A.npy', Full_Mean_A)
            np.save('Data/Full_Variance_A.npy', Full_Variance_A)

    Full_Hour = np.array([i % 24 for i in range(24*4)])
    Full_Sequence = np.arange(4*24)
    Full_CR = np.array([[7 + 24*i, 19 + 24*i] for i in range(4)])

    fig, axs = plt.subplots(2, 1, figsize = (12, 16))
    for i in range(len(LABELS)):
        if LABELS[i][1] == 'Pre': colors = 'black'
        else: colors = 'red'
        #colors = colors_name[i]
        axs[0].plot(Full_Sequence, Full_Mean_V[i], color = colors)
        axs[1].plot(Full_Sequence, Full_Variance_V[i], color = colors)
    axs[0].set_ylim(0,75)
    axs[1].set_ylim(0,12000)
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
    for i in range(len(LABELS)):
        if LABELS[i][1] == 'Pre': colors = 'black'
        else: colors = 'red'
        #colors = colors_name[i]
        axs[0].plot(Full_Sequence, Full_Mean_A[i], color = colors)
        axs[1].plot(Full_Sequence, Full_Variance_A[i], color = colors)
    axs[0].set_ylim(0,150)
    axs[1].set_ylim(0,100000)
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

def Display_HMM_Similar_States_Comparison(Mice, Wheel, Grooming, file_path):
    def Compare_Images(state_meaning, states):
        Speed = []
        Acce = []
        BodyL = []
        
        fig, axs = plt.subplots(1,len(Mice_N),figsize = (30, 8))
        for i in range(len(Mice_N)):
            state = states[Mice_N[i]] - 1
            Mouse = Mice[Mice_N[i]]

            Speed.append(Mouse.hmm.parameters[0].T[state][0])
            Acce.append(Mouse.hmm.parameters[0].T[state][1])
            BodyL.append(Mouse.hmm.parameters[0].T[state][2])
            
            if state_meaning == 'WheelMoving':
                for t in range(30, len(Mouse.arena.pellets.index)):
                    trigger = Mouse.arena.pellets.index[t]
                    start, end = trigger - pd.Timedelta("1S"), trigger
                    trigger_state = Mouse.mouse_pos.loc[start:end, 'state'][0]
                    if trigger_state == state and Mouse.mouse_pos.loc[start:end, 'r'][0] < Mouse.arena.radius:
                        if Mouse.mouse_pos.loc[start:end, 'state'][-1] == state:
                            break
                id = Mouse.arena.visits['start'].searchsorted(trigger, side='right') - 1
                patch = Mouse.arena.visits['patch'][id]
                
            if state_meaning == 'Grooming':
                patch_loc = [Mouse.arena.patch_location['Patch' + str(pt+1)] for pt in range(3)]
                starts, ends = [], []
                for j in range(1, len(Mouse.hmm.states)):
                    if Mouse.hmm.states[j-1] != state and Mouse.hmm.states[j] == state: starts.append(j)
                    if Mouse.hmm.states[j-1] == state and Mouse.hmm.states[j] != state: ends.append(j)
                for j in range(len(starts)):
                    if (ends[j] - starts[j]) >= 20 and Mouse.mouse_pos['r'][starts[j]] < Mouse.arena.radius:
                        x = Mouse.mouse_pos.smoothed_position_x[starts[j]]
                        y = Mouse.mouse_pos.smoothed_position_y[starts[j]]
                        patch_distance = [(x-patch_loc[p][0])**2 + (y-patch_loc[p][1])**2 for p in range(3)]
                        if min(patch_distance) < 30**2:
                            patch_idx = np.argsort(patch_distance)[0]
                            patch = 'Patch' + str(patch_idx + 1)
                            if Mouse.mouse_pos['r'][starts[j]] > np.sqrt((patch_loc[patch_idx][0]-Mouse.arena.origin[0])**2 + (patch_loc[patch_idx][1]-Mouse.arena.origin[1])**2):
                                start, end = Mouse.mouse_pos.index[starts[j]], Mouse.mouse_pos.index[ends[j]]
                                break
                

            if patch == 'Patch1':
                video_metadata = aeon.load(Mouse.root, social02.CameraPatch1.Video, start=start, end=end)
            if patch == 'Patch2':
                video_metadata = aeon.load(Mouse.root, social02.CameraPatch2.Video, start=start, end=end)
            if patch == 'Patch3':
                video_metadata = aeon.load(Mouse.root, social02.CameraPatch3.Video, start=start, end=end)
                
            video_metadata.index = video_metadata.index.round("20L")
            frames = video.frames(video_metadata)
            first_frame = next(frames)
            rgb_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            axs[i].imshow(rgb_frame)
            square_size = 0.2
            rect_x = 1 - square_size
            rect_y = 1 - square_size
            rect = patches.Rectangle((1 - square_size, 1 - square_size), square_size, square_size, 
                            transform=axs[i].transAxes, color=color_names[state])
            axs[i].add_patch(rect)
            axs[i].text(rect_x + square_size/2, rect_y + square_size/2, str(state+1),
                    transform=axs[i].transAxes, 
                    ha='center', va='center', 
                    color='white', fontweight='bold', fontsize=36)
            axs[i].set_title('Mouse=' + str(Mouse.mouse) + ', Phase=' + str(Mouse.type))
            axs[i].axis('off')
        plt.tight_layout()
        plt.savefig(file_path + state_meaning + '_Frames.png')

        print('Avegra Speed: ', np.mean(np.array(Speed))*2e-3, 'm/s')
        print('Avegrage Acce: ', np.mean(np.array(Acce))*2e-3, 'm/s2')
        print('Avegra Body Length: ', np.mean(np.array(BodyL))*2, 'mm')

    def Compare_KL_Divergence(state_meaning, states):
        def Exclude_Diagonal(matrix):
            return matrix[~np.eye(matrix.shape[0], dtype=bool)].reshape(matrix.shape[0], -1).reshape(-1)

        states_params = []
        all_kl_divergence = np.array([])
        for i in Mice_N:
            state = states[i]
            Mouse = Mice[i]
            
            states_params.append((Mouse.hmm.parameters[0].T[state-1], Mouse.hmm.parameters[2][state-1]))
            row, column = state - 1, state - 1
            all_kl_divergence = np.concatenate((all_kl_divergence,Exclude_Diagonal(Mouse.hmm.kl_divergence[row-1:row+2, column-1:column+2])))
        kl_divergence = create_kl_divergence_matrix(states_params)
        
        all_kl_matrix = np.log10(all_kl_divergence + 1)
        kl_matrix = np.log10(kl_divergence + 1)
        
        annot_array = np.array([[round(item, 2) for item in row] for row in kl_matrix])
        labels = ['$S_{' + str(i+1) + '}$' for i in Mice_N]
        fig, axs = plt.subplots(1,1, figsize=(10,8))
        sns.heatmap(kl_matrix, cmap='RdBu', ax = axs, square = 'True', cbar = True, vmin = 0, vmax = 6, annot=annot_array, annot_kws={'size': 14})
        axs.set_title("K-L Divergence Matrix", fontsize = 25)
        axs.set_xticklabels(labels)
        axs.set_yticklabels(labels, rotation = 0)
        axs.tick_params(axis='both', which='major', labelsize=20)
        plt.tight_layout()
        plt.savefig(file_path + state_meaning + '_KLMatrix.png')
        
        KL_divergence= Exclude_Diagonal(kl_matrix)
        print('Average Similar State KL: ', np.mean(KL_divergence))
        print('Average All KL: ', np.mean(all_kl_matrix))
        
        statistic, p_value = stats.ks_2samp(KL_divergence, all_kl_divergence)
        fig, axs = plt.subplots(1, 1, figsize=(12, 3))
        axs.hist(KL_divergence, bins = 10, color = 'red', alpha = 0.5, label = state_meaning)
        axs.hist(all_kl_matrix,bins = 10,  color = 'blue', alpha = 0.5, label = 'Other')
        axs.plot([],[], color = 'white', label = 'p = ' + str(round(p_value, 3)))
        axs.legend(loc = 'upper right', fontsize = 20)
        axs.set_xlabel('Distance', fontsize=20)
        axs.set_ylabel('Count.', fontsize=20)
        axs.tick_params(axis='both', which='major', labelsize=18)
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(file_path + state_meaning + 'KL_Compare.png')
        
        Compare_Images(state_meaning, states)
        
    
    if Wheel: 
        Compare_KL_Divergence(state_meaning = 'WheelMoving', states = [5,6,6,4,5,4,4,6])
    if Grooming: 
        Compare_KL_Divergence(state_meaning = 'Grooming', states = [4,4,5,5,6,3,3,3])
    
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
    
    fig, axs = plt.subplots(1,1,figsize = (10, 6))
    for i in range(len(Mice_N)):
        Mouse = Mice[Mice_N[i]]
        Visits = Mouse.arena.visits.dropna(subset=['speed'])
        Ends = Visits['end']
        
        STATES = Mouse.hmm.process_states.Event_Triggering(Mouse.mouse_pos, Ends, 3, 5, 'state', include_nan = 0)
        left_seconds = 3

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
        
        Duration_Mean.append(np.mean(Escapes))
        Duration_Var.append(np.std(Escapes))
    axs.bar(range(len(Duration_Mean)), Duration_Mean, yerr=Duration_Var, capsize=14)
    axs.set_ylabel('Duration (s)', fontsize = 30)
    axs.set_xticks(range(len(Duration_Mean)))
    axs.set_xticklabels([f'Session {i+1}' for i in Mice_N])
    axs.tick_params(axis='both', which='major', labelsize=16)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(file_path + 'EndVisitEscapeDuration_All.png')
    
    print(np.mean(ESCAPES))
    print(np.std(ESCAPES))
    
    print('Complete Display_Escape_Behaviour_Summary')
    
def Display_Distance_Comparison(file_path = '../Images/AllMice/'):
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
        axs[i].hist(pre, bins = 50, color = 'purple', alpha = 0.5, label = 'Pre')
        axs[i].hist(post,bins = 50,  color = 'green', alpha = 0.5, label = 'Post')
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
    
    '''-------------------------------Kinematics-------------------------------'''

    #Display_LDS_Parameters(Mice)

    #Display_Kinematics_Properties_Along_Time(Mice, file_path = file_path)
    
    #Display_Kinematics_Comparison(Mice, file_path = file_path)
    
    
    '''-------------------------------Behaviour-------------------------------'''
    #Display_Model_Selection(Mice, N = np.arange(3, 27), file_path = file_path)

    #Display_HMM_States_Duration_Along_Time(Mice, file_path = file_path) 
    
    Display_HMM_Similar_States_Comparison(Mice, Wheel = False, Grooming = True, file_path = file_path)
    
    #Display_Escape_Behaviour_Summary(Mice, file_path = file_path)
    
    #Display_Distance_Comparison(file_path = file_path)
        
    '''-------------------------------Regression-------------------------------'''
    #Display_Visit_Prediction(Mice, model = 'linear', file_path = '../Images/Social_Regression/All-Mice/', title = 'Linear_Regression.png')                                            
    #Display_Visit_Prediction(Mice, model = 'MLP', file_path = '../Images/Social_Regression/All-Mice/', title = 'MLP.png')

if __name__ == "__main__":
        main()
        
        
        
        