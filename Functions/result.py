import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import pandas as pd
import seaborn as sns

import statsmodels.api as sm
import scipy.stats as stats
import random
from scipy.linalg import inv, det

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

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

color_names = [
    'black', 'blue', 'red', 'tan', 'green', 'brown', 
    'purple', 'orange', 'magenta', 'olive', 'pink', 
    'darkblue', 'lime', 'cyan', 'turquoise', 'gold', 
    'navy', 'maroon', 'teal', 'grey']

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


def Exclude_Diagonal(matrix):
    return matrix[~np.eye(matrix.shape[0], dtype=bool)].reshape(matrix.shape[0], -1).reshape(-1)

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

class HMM:
    def __init__(self, Mouse):
        self.Mouse = Mouse 
        self.Mouse_title = Mouse.type + '_' + Mouse.mouse
        self.N = Mouse.hmm.n_state
        self.TransM = Mouse.hmm.TransM
        self.kl_matrix = np.log10(Mouse.hmm.kl_divergence + 1)
        self.connecM = Mouse.hmm.ConnecM
        self.Mouse.mouse_pos['state'] = pd.Series(Mouse.hmm.states, index = Mouse.mouse_pos.index)
        
        self.Pellets = Mouse.arena.pellets.index
        self.Visits = Mouse.arena.visits.dropna(subset=['speed'])
        self.Starts = self.Visits['start'].to_numpy()
        self.Ends = self.Visits['end'].to_numpy()
        self.Entry = Mouse.arena.entry 

        self.color_names = ['black', 'blue', 'red', 'tan', 'green', 'brown', 
                            'purple', 'orange', 'magenta', 'olive', 'pink', 
                            'darkblue', 'lime', 'cyan', 'turquoise', 'gold', 
                            'navy', 'maroon', 'teal', 'grey']
        self.colors = sns.xkcd_palette(self.color_names[0:self.N])
        self.cmap = gradient_cmap(self.colors)
        

    class Model_Features:
        def __init__(self, HMM):
            self.Mouse = HMM.Mouse
            self.Mouse_title = HMM.Mouse_title
            self.HMM = HMM
            self.feature_label = ['SPEED (m/s)', 'ACCELERATION (m/s$^2$)', 'BODY LENGTH (mm)']
            self.file_path = None
            self.print = False
    
        def PlotFeatures_Zoom(self):
            FEATURE = self.feature_label
            scale = 2e-3
            N = self.Mouse.hmm.n_state
            N_zoom_state = 7
            
            Params = self.Mouse.hmm.parameters
            
            fig, axs = plt.subplots(len(FEATURE), 2, figsize = (N + N_zoom_state, len(FEATURE)*7-1), gridspec_kw={'width_ratios': [N, N_zoom_state]})
            for i in range(len(FEATURE)):
                if i > 1: scale = 2
                axs[i][0].bar(range(N), Params[0][i]*scale, yerr=1.65 * (Params[1][i] **0.5) * scale, capsize=14)
                axs[i][0].set_xticks(range(0, N), [str(j+1) for j in range(N)])
                axs[i][1].bar(range(N_zoom_state), Params[0][i][:N_zoom_state]*scale, yerr=1.65 * (Params[1][i][:N_zoom_state] ** 0.5 )* scale, capsize=14)
                axs[i][1].set_xticks(range(0, N_zoom_state), [str(j+1) for j in range(N_zoom_state)])

            for i in range(len(FEATURE)):
                for j in range(2):
                    axs[i][j].set_ylabel(FEATURE[i], fontsize = 35)
                    axs[i][j].spines['top'].set_visible(False)
                    axs[i][j].spines['right'].set_visible(False)
                    axs[i][j].tick_params(axis='both', which='major', labelsize=30)
                    axs[2][j].set_xlabel('State', fontsize = 40)
            plt.tight_layout()
            plt.savefig(self.file_path + 'Parameter/' + self.Mouse_title + '_Zoom.png')
            if self.print: plt.show()
            plt.close()  
        
        def PlotFeatures(self):
            FEATURE = self.feature_label
            scale = 2e-3
            N = self.Mouse.hmm.n_state
            Params = self.Mouse.hmm.parameters
            
            fig, axs = plt.subplots(len(FEATURE), 1, figsize = (N, len(FEATURE)*7-1))
            for i in range(len(FEATURE)):
                if i > 1: scale = 2
                axs[i].bar(range(N), Params[0][i]*scale, yerr=1.65 * (Params[1][i] **0.5) * scale, capsize=14)
                axs[i].set_xticks(range(0, N), [str(j+1) for j in range(N)])
                axs[i].set_ylabel(FEATURE[i], fontsize = 35)
                axs[i].spines['top'].set_visible(False)
                axs[i].spines['right'].set_visible(False)
                axs[i].tick_params(axis='both', which='major', labelsize=30)
                axs[2].set_xlabel('State', fontsize = 40)
            plt.tight_layout()
            plt.savefig(self.file_path + 'Parameter/' + self.Mouse_title + '.png')
            if self.print: plt.show()
            plt.close()     
        
        def PlotMatrix(self, matrix, digit, matrix_title, sub_file_path):
            annot_array = np.array([[round(item, digit) for item in row] for row in matrix])
            labels = ['$S_{' + str(i+1) + '}$' for i in range(len(matrix))]
            
            fig, axs = plt.subplots(1,1, figsize=(10,8))
            sns.heatmap(matrix, cmap='RdBu', ax = axs, square = 'True', cbar = True, annot=annot_array, annot_kws={'size': 14})
            axs.set_title(matrix_title, fontsize = 25)
            axs.set_xticklabels(labels)
            axs.set_yticklabels(labels, rotation = 0)
            axs.tick_params(axis='both', which='major', labelsize=20)
            plt.tight_layout()
            plt.savefig(self.file_path + sub_file_path +self.Mouse_title + '.png')
            if self.print: plt.show()
            plt.close()
        
        def Display(self, Parameter, Parameter_zoom, TransM, ConnecM, K_L, file_path, print):
            self.file_path = file_path
            self.print = print 
            
            if Parameter: self.PlotFeatures()
            if Parameter_zoom: self.PlotFeatures_Zoom()
            if TransM: self.PlotMatrix(self.HMM.TransM, 3, "Transition Matrix", 'TransM/')
            if ConnecM: self.PlotMatrix(self.HMM.connecM, 3, "Transition Matrix (Diag. Exclud.)", 'ConnecM/')
            if K_L: self.PlotMatrix(self.HMM.kl_matrix, 1, "(Log-scale) KL-Divergence", 'K_L/')
    
    class States_Features:
        def __init__(self, HMM):
            self.HMM = HMM
            self.Mouse = HMM.Mouse
            self.Mouse_title = HMM.Mouse_title
            self.file_path = None
            self.print = False

        def PlotPosition(self):
            N = self.HMM.N
            states = self.Mouse.hmm.states
            mouse_pos = self.Mouse.mouse_pos
            x, y= [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)]
            for i in range(N):
                x[i] = mouse_pos['smoothed_position_x'][states==i]
                y[i] = mouse_pos['smoothed_position_y'][states==i]
                
            fig, axs = plt.subplots(2, 5, figsize = (10*8-2,24))
            axs = axs.flatten()
            for i in range(N):
                axs[i].scatter(x[i], y[i], color = self.HMM.color_names[i], s = 0.001, alpha = 1)
                axs[i].set_xlim((100,1400))
                axs[i].set_ylim((-20,1100))
                axs[i].set_title('State' + str(i+1), fontsize = 32)
                axs[i].set_xlabel('X (px)', fontsize = 30)
                axs[i].set_ylabel('Y (px)', fontsize = 30)
                axs[i].spines['top'].set_visible(False)
                axs[i].spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(self.file_path + 'Position/' + self.Mouse_title + '.png')
            if self.print: plt.show()
            plt.close()

        def PlotDurationInPlaces(self):
            N = self.Mouse.hmm.n_state
            states = self.Mouse.hmm.states
            mouse_pos = self.Mouse.mouse_pos
            r = [np.array([]) for _ in range(N)]
            for i in range(N): r[i] = mouse_pos['r'][states == i]
                
            fig, axs = plt.subplots(2, 5, figsize = (10*8-2,12))
            axs = axs.flatten()
            for i in range(N):
                num_nest = len(r[i][r[i] > int(self.Mouse.arena.metadata.ActiveRegion.ArenaOuterRadius)]) / len(r[i])
                num_arena = len(r[i][r[i] < int(self.Mouse.arena.metadata.ActiveRegion.ArenaInnerRadius)]) / len(r[i])
                num_corridor = 1 - num_nest - num_arena
                
                axs[i].bar(range(3), [num_nest,num_corridor, num_arena], color=['blue', 'red', 'black'])
                axs[i].set_xticks(range(0, 3), ['Nest', 'Arena', 'Corrid.'])
                axs[i].set_yticks([0,1])
                axs[i].spines['top'].set_visible(False)
                axs[i].spines['right'].set_visible(False)
                axs[i].tick_params(axis='both', which='major', labelsize=60)

            plt.tight_layout()
            plt.savefig(self.file_path + 'Position/' + self.Mouse_title + '_Duration.png')
            if self.print: plt.show()
            plt.close()
        
        def PlotFrequencyAlongTime(self):
            N = self.HMM.N
            states = self.Mouse.hmm.states
            mouse_pos = self.Mouse.mouse_pos
            
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
            plt.savefig(self.file_path + 'Frequency/' + self.Mouse_title + '.png')
            if self.print: plt.show()
            plt.close()
        
        def PlotStatesAlongTime(self):
            start, end = self.Mouse.active_chunk[0], self.Mouse.active_chunk[1]
            mouse_pos = self.Mouse.mouse_pos[start:end]
            
            states_prob = self.Mouse.hmm.process_states.State_Dominance(mouse_pos, time_seconds = 10)
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

            N = self.HMM.N
            fig, axs = plt.subplots(N, 1, figsize=(50, 4*N-1))
            for i in range(N):
                states_prob[i].plot(color = self.HMM.color_names[i], ax = axs[i])
                for t in range(len(START)):
                    axs[i].axvspan(START[t],END[t], color='lightblue', alpha=0.5)
                axs[i].spines['top'].set_visible(False)
                axs[i].spines['right'].set_visible(False)
            plt.savefig(self.file_path + 'Example/' + self.Mouse_title + '.png')
            if self.print: plt.show()
            plt.close()
        
        def Display(self, Position, Duration, Frequency, Example, file_path, print = False):
            self.file_path = file_path
            self.print = print 
            
            if Position: self.PlotPosition()
            if Duration: self.PlotDurationInPlaces()
            if Frequency: self.PlotFrequencyAlongTime()
            if Example: self.PlotStatesAlongTime()
            
    class Characterize_Timepoints:
        def __init__(self, HMM, event_name, Events, left_seconds, right_seconds):
            self.HMM = HMM
            self.Mouse = HMM.Mouse
            self.Mouse_title = HMM.Mouse_title
            self.N = HMM.N 
            
            self.event_name = event_name 
            self.Events = Events 
            self.left_seconds = left_seconds
            self.right_seconds = right_seconds
            self.file_path = None 
            self.file_name = None
            self.print = False
            
            self.Valid_Events, self.STATES_with_nan = None, None
            self.STATES = None
            self.plot_sequence, self.dominant_sequence, self.comparable_sequence = None, None, None
            self.event_distance = []
            
            self.Compare_State_Sequence()
            
        def Compare_State_Sequence(self):
            self.Valid_Events, self.STATES_with_nan = self.Mouse.hmm.process_states.Event_Triggering(self.Mouse.mouse_pos, self.Events, self.left_seconds, self.right_seconds, 'state', insert_nan = 1, return_Events = True)
            self.STATES = np.delete(self.STATES_with_nan, int(self.left_seconds*10),axis = 1)

            start = int((self.left_seconds-1.5) * 10)
            end = int((self.left_seconds+1.5) * 10)
            self.plot_sequence = self.Mouse.hmm.process_states.Find_Event_Sequence(self.STATES_with_nan, penalty = 0)
            self.dominant_sequence = self.Mouse.hmm.process_states.Find_Event_Sequence(self.STATES, penalty = 0)
            self.comparable_sequence = self.dominant_sequence[start:end]
            
            for i in range(len(self.STATES)):
                state = self.STATES[i][start:end]
                self.event_distance.append(self.Mouse.hmm.process_states.Compare_Sequence(state, self.comparable_sequence, self.HMM.kl_matrix))

        def EventHeatmap(self):
            STATES, sequence = self.STATES_with_nan, self.plot_sequence
            left_seconds, right_seconds = self.left_seconds, self.right_seconds
            fig, axs = plt.subplots(2, 1, figsize=(10, 10.5),gridspec_kw={'height_ratios': [10, 0.5]})
            sns.heatmap(STATES, cmap=self.HMM.cmap, ax=axs[0], vmin=0, vmax = self.N-1, cbar = False)
            sns.heatmap(np.array([sequence]),cmap=self.HMM.cmap, ax=axs[1], vmin=0, vmax = self.N-1, cbar = False)
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
                if np.isnan(state): continue
                state_length = flags[i+1] - flags[i]
                if state_length < 3 :continue
                center_position = (flags[i+1] + flags[i])/2
                axs[1].text(center_position, 0.5, f'{int(state+1)}', 
                            ha='center', va='center', fontsize=24, 
                            fontweight='bold', color='white',
                            transform=axs[1].transData)

            fig.suptitle(self.event_name, fontsize=36, y=0.98)
            axs[0].text(0, 1.0, f't=-{left_seconds}s', fontsize=20, ha='left', va='bottom', transform=axs[0].transAxes)
            axs[0].text(1.0*left_seconds/(left_seconds + right_seconds + 0.1), 1.0, f't=0s', fontsize=20, ha='center', va='bottom', transform=axs[0].transAxes)
            axs[0].text(1, 1.0, f't=+{right_seconds}s', fontsize=20, ha='right', va='bottom', transform=axs[0].transAxes)
            plt.tight_layout()
            plt.savefig(self.file_path + self.file_name + '/Heatmap_' + self.Mouse_title + '.png')
            if self.print: plt.show()
            plt.close()
        
        def EventSequenceDistance(self, STATES_Random):
            
            event_distance_Shuffle = []
            start = int((self.left_seconds-1.5) * 10)
            end = int((self.left_seconds+1.5) * 10)
            for i in range(len(STATES_Random)):
                state_ = STATES_Random[i][start:end]
                event_distance_Shuffle.append(self.Mouse.hmm.process_states.Compare_Sequence(state_, self.comparable_sequence, self.HMM.kl_matrix))

            fig, axs = plt.subplots(1,1,figsize = (3, 10))
            axs.plot(self.event_distance, np.arange(len(self.STATES),0,-1),color = 'red')
            axs.set_xlim((0,50))
            axs.spines['top'].set_visible(False)
            axs.spines['right'].set_visible(False)
            axs.set_ylabel('#Events', fontsize = 20)
            axs.set_xlabel('Distance', fontsize = 22)
            axs.set_xticks([0,25,50])
            axs.set_xticklabels(['0','25','50'])
            axs.set_yticks([0,len(self.event_distance)])
            axs.set_yticklabels(['Late','Early'])
            axs.tick_params(axis='both', which='major', labelsize=14)
            plt.tight_layout()
            plt.savefig(self.file_path + self.file_name + '/SequenceDistance_' + self.Mouse_title + '.png')
            if self.print: plt.show()
            plt.close()


            statistic, p_value = stats.ks_2samp(np.array(self.event_distance), np.array(event_distance_Shuffle))
            fig, axs = plt.subplots(1, 1, figsize=(12, 3))
            axs.hist(self.event_distance, bins = 30, color = 'red', alpha = 0.5, label = 'Original')
            axs.hist(event_distance_Shuffle,bins = 30,  color = 'blue', alpha = 0.5, label = 'Shuffled')
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
            plt.savefig(self.file_path + self.file_name + '/SequenceDistance_Validation_' + self.Mouse_title + '.png')
            if self.print: plt.show()
            plt.close()

        def EventSequenceDivergence(self):
            left_seconds, right_seconds = self.left_seconds, self.right_seconds
            Distance = np.zeros((len(self.STATES), len(self.STATES[0])))
            for s in range(len(self.STATES)):
                for j in range(len(self.STATES[s])):
                    Distance[s,j] = self.HMM.kl_matrix[int(self.STATES[s][j])][int(self.sequence[j])]
            
            STATES = self.STATES
            sec = int(len(STATES)/4)
            
            fig, axs = plt.subplots(1,1,figsize = (10, 8))
            axs.plot(np.arange(len(STATES[0])),np.mean(Distance[:sec], axis = 0), color = 'brown', alpha = 1, label = 'Early')
            axs.plot(np.arange(len(STATES[0])),np.mean(Distance[sec:-sec], axis = 0), color = 'green', alpha = 1, label = 'Mid')
            axs.plot(np.arange(len(STATES[0])),np.mean(Distance[-sec:], axis = 0), color = 'blue', alpha = 1, label = 'Late')
            axs.axvline(x = left_seconds*10, linestyle = '--', color = 'red')
            axs.legend(fontsize = 20)
            axs.spines['top'].set_visible(False)
            axs.spines['right'].set_visible(False)
            axs.set_ylabel('(Log) K-L Divergence', fontsize = 30)
            axs.set_xlabel('Time (s)', fontsize = 30)
            axs.set_xticks([0,left_seconds*10,(left_seconds+right_seconds)*10-1])
            axs.set_xticklabels(['-' + str(left_seconds),'0','+' + str(right_seconds)])
            axs.tick_params(axis='both', which='major', labelsize=20)
            plt.tight_layout()
            plt.savefig(self.file_path + self.file_name + '/StateDistance_' + self.Mouse_title + '.png')
            if self.print: plt.show()
            plt.close()
        
        def EventPosition(self, STATES, X, Y):
            left_seconds, right_seconds = self.left_seconds, self.right_seconds
            start = int((left_seconds-1.5) * 10)
            end = int((left_seconds+1.5) * 10)
            STATES, X, Y = STATES[:, start:end], X[:, start:end], Y[:, start:end]
            fig, axs = plt.subplots(1, 1, figsize=(20, 20))
            for i in range(len(STATES)):
                colors = np.array(color_names)[STATES[i].astype(int)]
                x = X[i]
                y = Y[i]
                dx = np.diff(x)
                dy = np.diff(y)
                axs.quiver(x[:-1], y[:-1], dx, dy, color=colors[:-1], 
                        angles='xy', scale_units='xy', scale=1.1, 
                        width=0.001, headwidth=3, headlength=3)
            axs.set_aspect('equal')
            axs.set_xlim((100,1400))
            axs.set_ylim((-20,1100))
            plt.savefig(self.file_path + self.file_name + '/Position_' + self.Mouse_title + '.png')
            if self.print: plt.show()
            plt.close()
        
        def Display(self, Heatmap, Position, SequenceDivergence, SequenceDistance, file_path, file_name, print = False):
            self.file_path = file_path
            self.file_name = file_name
            self.print = print 
            
            mouse_pos = self.Mouse.mouse_pos
            
            if Heatmap: self.EventHeatmap()

            if SequenceDivergence: self.EventSequenceDivergence()
            if SequenceDistance: 
                idx = random.sample(range(len(self.Mouse.mouse_pos)-100), len(self.STATES))
                random_events = self.Mouse.mouse_pos.index[idx]
                STATES_Random = self.Mouse.hmm.process_states.Event_Triggering(self.Mouse.mouse_pos, random_events, self.left_seconds, self.right_seconds, 'state', insert_nan = 0)
                self.EventSequenceDistance(STATES_Random)
            
            if Position:
                X = self. Mouse.hmm.process_states.Event_Triggering(mouse_pos, self.Events, self.left_seconds, self.right_seconds, 'smoothed_position_x', insert_nan = 0)
                Y = self.Mouse.hmm.process_states.Event_Triggering(mouse_pos, self.Events, self.left_seconds, self.right_seconds, 'smoothed_position_y', insert_nan = 0)
                self.EventPosition(self.STATES, X, Y)
                
    class EventPrediction:
        def __init__(self, HMM, model, file_path, print = False):
            self.Mouse = HMM.Mouse
            self.Mouse_title = HMM.Mouse_title
            self.HMM = HMM
            self.model = model
            self.file_path = file_path
            self.print = print
        
class Behaviour:
    def __init__(self, Mouse):
        self.Mouse = Mouse 
        self.Mouse_title = Mouse.type + '_' + Mouse.mouse
        self.Mouse.mouse_pos['state'] = pd.Series(Mouse.hmm.states, index = Mouse.mouse_pos.index)
        
        self.Pellets = Mouse.arena.pellets.index
        self.Visits = Mouse.arena.visits.dropna(subset=['speed'])
        self.Starts = self.Visits['start']
        self.Ends = self.Visits['end']
        self.Entry = Mouse.arena.entry 

    class Gate_to_Patch:
        def __init__(self, Behaviour):
            self.Mouse = Behaviour.Mouse
            self.Mouse_title = Behaviour.Mouse_title
            self.Visits = Behaviour.Visits
            self.file_path = None
            self.print = False
            
        def PlotPositionMap(self):
            fig, axs = plt.subplots(1, 1, figsize=(8,8))
            # Draw inner circle
            origin = self.Mouse.arena.origin 
            radius = int(self.Mouse.arena.metadata.ActiveRegion.ArenaInnerRadius)
            angles = np.linspace(0, 2 * np.pi, 1000)

            x = origin[0] + radius * np.cos(angles)
            y = origin[1] + radius * np.sin(angles)
            axs.scatter(x, y, color='grey', s=2)  

            # Draw outer circle
            radius = int(self.Mouse.arena.metadata.ActiveRegion.ArenaOuterRadius)
            x = origin[0] + radius * np.cos(angles)
            y = origin[1] + radius * np.sin(angles)
            axs.scatter(x, y, color='grey', s=2)  

            for i in range(len(self.Visits)):
                mouse_pos_sub =self.Mouse.mouse_pos[self.Visits['start'].to_numpy()[i] - pd.Timedelta(str(self.Visits['entry'].to_numpy()[i]) + 'S'):self.Visits['start'].to_numpy()[i]]
                colors = ['red', 'blue']
                x = mouse_pos_sub['smoothed_position_x'].to_numpy()[[0,-1]]
                y = mouse_pos_sub['smoothed_position_y'].to_numpy()[[0,-1]]
                axs.scatter(x, y, color = colors, s=6)
            axs.legend(loc = 'lower left', fontsize = 12)
            axs.set_xlabel('X', fontsize = 16)
            axs.set_ylabel('Y', fontsize = 16)
            axs.tick_params(axis='both', which='major', labelsize=12)
            axs.spines['top'].set_visible(False)
            axs.spines['right'].set_visible(False)
            axs.set_aspect('equal')
            axs.set_xlim((100,1400))
            axs.set_ylim((-20,1100))
            plt.tight_layout()
            plt.savefig(self.file_path + 'Position/' + self.Mouse_title + '.png')
            if self.print: plt.show()
            plt.close()
        
        def PlotDuration(self):
            start, end = self.Mouse.mouse_pos.index[0], self.Mouse.mouse_pos.index[-1]
            colors = ['red', 'blue', 'green']
            fig, axs = plt.subplots(3, 1, figsize=(30,15))
            for i in range(3):
                Visits_sub = self.Visits[self.Visits['patch'] == 'Patch' + str(i+1)]
                axs[i].scatter(Visits_sub.start, Visits_sub.entry, color = colors[i])
                axs[i].set_xlim((start, end))
                axs[i].set_xlabel('Start Time', fontsize = 16)
                axs[i].set_ylabel('Duration From Enter Arena (s)', fontsize = 16)
                axs[i].tick_params(axis='both', which='major', labelsize=12)
                axs[i].spines['top'].set_visible(False)
                axs[i].spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(self.file_path + 'Duration/' +  self.Mouse_title + '.png')
            if self.print: plt.show()
            plt.close()
            
        def PlotDistance(self):
            start, end = self.Mouse.mouse_pos.index[0], self.Mouse.mouse_pos.index[-1]
            
            patch_loc = [self.Mouse.arena.patch_location['Patch' + str(pt+1)] for pt in range(3)]
            gate = [self.Mouse.arena.origin[0] - self.Mouse.arena.radius, self.Mouse.arena.origin[1]]
            patch_gate_distance = [np.sqrt((patch_loc[pt][0] - gate[0])**2 + (patch_loc[pt][1] - gate[1])**2)*2e-3 for pt in range(3)]
            
            colors = ['red', 'blue', 'green']
            fig, axs = plt.subplots(3, 1, figsize=(30,15))
            
            for i in range(3):
                Visits_sub = self.Visits[self.Visits['patch'] == 'Patch' + str(i+1)]
                distance = []
                for j in range(len(Visits_sub)):
                    mouse_pos_sub = self.Mouse.mouse_pos[Visits_sub['start'].to_numpy()[j] - pd.Timedelta(str(Visits_sub['entry'].to_numpy()[j]) + 'S'):Visits_sub['start'].to_numpy()[j]]
                    time_diff = mouse_pos_sub.index.to_series().diff().dt.total_seconds()
                    distance_x = np.abs(mouse_pos_sub['smoothed_velocity_x'] * time_diff).sum()
                    distance_y = np.abs(mouse_pos_sub['smoothed_velocity_y'] * time_diff).sum()
                    total_distance = np.sqrt(distance_x**2 + distance_y**2)
                    distance.append(total_distance*2e-3)
                distance = np.array(distance)

                axs[i].scatter(Visits_sub.start, distance, color = colors[i])
                axs[i].axhline(y=patch_gate_distance[i], color = 'black', linestyle = '--')
                axs[i].set_xlim((start, end))
                axs[i].set_xlabel('Start Time', fontsize = 16)
                axs[i].set_ylabel('Distance From Enter Arena (m)', fontsize = 16)
                axs[i].tick_params(axis='both', which='major', labelsize=12)
                axs[i].spines['top'].set_visible(False)
                axs[i].spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(self.file_path + 'Distance/' +  self.Mouse_title + '.png')
            if self.print: plt.show()
            plt.close()
            
        def Display(self, Position, Duration, Distance, file_path, print = False):
            self.file_path = file_path 
            self.print = print 
            
            Visits = self.Visits
            first_visit = [1]
            for i in range(1, len(Visits)):
                if abs((Visits['start'][i] - Visits['start'][i-1]).total_seconds() - (Visits['entry'][i] - Visits['entry'][i-1])) < 1:
                    first_visit.append(0)
                else: first_visit.append(1)
            Visits.loc[Visits.index, 'first_visit'] = np.array(first_visit)
            self.Visits = Visits[Visits['first_visit'] == 1]

            if Position: self.PlotPositionMap()
            if Duration: self.PlotDuration()
            if Distance: self.PlotDistance()
            
    class Escape_after_Forage:
        def __init__(self, Behaviour):
            self.Mouse = Behaviour.Mouse
            self.Mouse_title = Behaviour.Mouse_title
            self.Visits = Behaviour.Visits
            self.file_path = None
            self.print = False
            
            self.escape_times = self.Get_Escape_Times()
            self.escape_time_mean = np.mean(self.escape_times)
            self.escape_time_std = np.std(self.escape_times)
        
        def Get_Escape_Times(self):
            Ends = self.Visits['end'].to_numpy()
        
            left_seconds, right_seconds = 3, 5
            STATES = self.Mouse.hmm.process_states.Event_Triggering(self.Mouse.mouse_pos, Ends, left_seconds, right_seconds, 'state', insert_nan = 0)
            
            Escapes = np.zeros(len(STATES))
            for i in range(len(STATES)):
                state = STATES[i]
                escape_state = state[left_seconds*10]
                finish_escape = left_seconds*10
                for j in range(int(left_seconds*10), len(state)-1):
                    if state[j] == escape_state and state[j+1]!=escape_state: 
                        finish_escape = j + 1
                        break
                Escapes[i] = (finish_escape - left_seconds*10)/10
            return Escapes

class Kinematics:
    def __init__(self, Mouse):
        self.Mouse = Mouse
        self.Mouse_title = Mouse.type + '_' + Mouse.mouse
        
        self.starts, self.ends = self.Get_Hour_Sessions()
        self.Full_Hour = np.array([i % 24 for i in range(24*4)])
        self.Full_Sequence = np.arange(4*24)
        self.Full_CR = np.array([[7 + 24*i, 19 + 24*i] for i in range(4)])
        self.Mean_V = np.full(24*4, np.nan)
        self.Mean_A = np.full(24*4, np.nan)
        self.Variance_V = np.full(24*4, np.nan)
        self.Variance_A = np.full(24*4, np.nan)
        self.Get_Kinematics_Properties()
        
    
    def Get_Hour_Sessions(self):
        start, end = self.Mouse.mouse_pos.index[0], self.Mouse.mouse_pos.index[-1]
        starts, ends = [],[]
        while start < end:
            if start.minute != 0:
                end_ = pd.Timestamp(year = start.year, month = start.month, day = start.day, hour = start.hour+1, minute=0, second=0)
            else: 
                end_ = start + pd.Timedelta('1H')

            starts.append(start)
            ends.append(end_)
            start = end_  
        return starts, ends
        
    def Get_Kinematics_Properties(self):
        day = 0
        for i in range(len(self.starts)):
            hour = self.starts[i].hour
            if hour == 0: day += 1
            
            index_in_full_sequence = day*24 + hour
            
            df = self.Mouse.mouse_pos[self.starts[i]:self.ends[i]]
            if len(df) == 0: continue
            
            speed = df.smoothed_speed
            self.Mean_V[index_in_full_sequence] = np.mean(speed)
            self.Variance_V[index_in_full_sequence] = np.var(speed)
                
            acce = df.smoothed_acceleration
            self.Mean_A[index_in_full_sequence] = np.mean(acce)
            self.Variance_A[index_in_full_sequence] = np.var(speed)
    
    class kinematics:
        def __init__(self, Kinematics):
            self.Mouse = Kinematics.Mouse
            self.file_path = None
            self.print = False
            self.Mouse_title = Kinematics.Mouse_title
            self.Kinematics = Kinematics
            
        def PlotTrace(self):
            mouse_pos = self.Mouse.mouse_pos
            
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
            plt.savefig(self.file_path + 'Trace/' + self.Mouse_title+'.png')
            if self.print: plt.show()
            plt.close() 
            
        def PlotProperty(self):
            fig, axs = plt.subplots(2, 2, figsize = (24, 16))
            axs[0][0].plot(self.Kinematics.Full_Sequence, self.Kinematics.Mean_V)
            axs[0][0].set_ylim(0,150)
            axs[0][0].set_ylabel('Mean', fontsize = 40)
            axs[1][0].plot(self.Kinematics.Full_Sequence, self.Kinematics.Variance_V)
            axs[1][0].set_ylim(0,14000)
            axs[1][0].set_ylabel('Variance', fontsize = 40)
            axs[1][0].set_xlabel('Hours', fontsize = 40)
            axs[0][0].set_title('Speed', fontsize = 45)

            axs[0][1].plot(self.Kinematics.Full_Sequence, self.Kinematics.Mean_A)
            axs[0][1].set_ylabel('Mean', fontsize = 40)
            axs[0][1].set_ylim(0,200)
            axs[1][1].plot(self.Kinematics.Full_Sequence, self.Kinematics.Variance_A)
            axs[1][1].set_ylim(0,14000)
            axs[1][1].set_ylabel('Variance', fontsize = 40)
            axs[1][1].set_xlabel('Hours', fontsize = 40)
            axs[0][1].set_title('Acceleration', fontsize = 45)

            for i in range(2):
                for j in range(2):
                    axs[i][j].set_xticks(self.Kinematics.Full_Sequence[::6], self.Kinematics.Full_Hour[::6])
                    axs[i][j].tick_params(axis='both', which='major', labelsize=25)
                    for t in range(len(self.Kinematics.Full_CR)):
                        axs[i][j].axvspan(self.Kinematics.Full_CR[t][0],self.Kinematics.Full_CR[t][1], color='lightblue', alpha=0.5)
                    axs[i][j].spines['top'].set_visible(False)
                    axs[i][j].spines['right'].set_visible(False)

            plt.tight_layout()
            plt.savefig(self.file_path + 'Property/' + self.Mouse_title+'.png')
            if self.print: plt.show()
            plt.close() 
            
            
        def Display(self, Trace, Property, file_path, print = False):
            self.file_path = file_path
            self.print = print
            if Trace: self.PlotTrace()
            if Property: self.PlotProperty()
        
class Summary:
    def __init__(self, Mice):
        self.Mice = Mice
        self.types = [Mice[i].type for i in range(len(Mice))]
        self.names = [Mice[i].mouse for i in range(len(Mice))]
        self.Mice_N = np.array([0,1,3,4,5,6,7])
        self.KL_M = self.Construct_KL_Matrix_Across_All_Sessions()
        
    def Construct_KL_Matrix_Across_All_Sessions(self):
        M = np.zeros((len(self.Mice_N), len(self.Mice_N), 10, 10))
        for i in range(len(self.Mice_N)):
            for j in range(len(self.Mice_N)):
                for s1 in range(10):
                    for s2 in range(10):
                        p_means, p_variances = (self.Mice[self.Mice_N[i]].hmm.parameters[0].T[s1], self.Mice[self.Mice_N[i]].hmm.parameters[2][s1])
                        q_means, q_variances = (self.Mice[self.Mice_N[j]].hmm.parameters[0].T[s2], self.Mice[self.Mice_N[j]].hmm.parameters[2][s2])
                        M[i,j,s1, s2] = kl_divergence_gaussian(p_means, p_variances, q_means, q_variances)
        return M 
    
    class Single_State:
        def __init__(self, Summary, state_meaning, states):
            self.state_meaning = state_meaning 
            self.states = states
            self.file_path = None
            self.print = False 
            
            self.Mice = Summary.Mice
            self.Mice_N = Summary.Mice_N
            
        def PlotFeature(self):
            states_params = []
            for i in self.Mice_N:
                state = self.states[i]
                Mouse = self.Mice[i]
                states_params.append((Mouse.hmm.parameters[0].T[state-1],Mouse.hmm.parameters[1].T[state-1]))
            states_params = np.array(states_params).T
            
            feature = ['smoothed_speed', 'smoothed_acceleration', 'head-spine3']
            FEATURE = ['SPEED (m/s)', 'ACCELERATION (m/s$^2$)', 'BODY LENGTH (mm)']
            indices = [Mouse.hmm.features.index(element) for element in feature]
            
            scale = 2e-3
            fig, axs = plt.subplots(3, 1, figsize = (len(self.Mice_N), 3*7-1))
            for i in range(3):
                if i > 1: scale = 2
                axs[i].bar(range(len(self.Mice_N)), states_params[indices[i]][0] * scale, yerr = 1.65 * (states_params[indices[i]][1] ** 0.5) * scale, capsize=14)
                axs[i].set_xticks(range(0, len(self.Mice_N)), [str(j+1) for j in self.Mice_N])

            axs[0].set_yticks([0, 0.005, 0.010])
            axs[1].set_yticks([0, 0.01, 0.02, 0.03, 0.04])
            axs[2].set_yticks([0, 10, 20, 30, 40, 50, 60])

            for i in range(3):
                axs[i].set_ylabel(FEATURE[i], fontsize = 35)
                axs[i].spines['top'].set_visible(False)
                axs[i].spines['right'].set_visible(False)
                axs[i].tick_params(axis='both', which='major', labelsize=30)
                axs[2].set_xlabel('Session', fontsize = 40)
            plt.tight_layout()
            plt.savefig(self.file_path + self.state_meaning + '_Feature_Compare.png')  
            if self.print: plt.show()
            plt.close()
        
        def PlotKLDivergence(self):
            states_params = []
            all_kl_divergence = []
            for i in self.Mice_N:
                state = self.states[i] - 1
                states_params.append((self.Mice[i].hmm.parameters[0].T[state], self.Mice[i].hmm.parameters[2][state]))
                for j in self.Mice_N:
                    if i == j: continue
                    state_ = self.states[j] - 1
                    for s in range(self.Mice[j].hmm.n_state):
                        if s == state_: continue
                        all_kl_divergence.append(kl_divergence_gaussian(self.Mice[i].hmm.parameters[0].T[state], self.Mice[i].hmm.parameters[2][state], self.Mice[j].hmm.parameters[0].T[s], self.Mice[j].hmm.parameters[2][s]))
                        all_kl_divergence.append(kl_divergence_gaussian(self.Mice[j].hmm.parameters[0].T[s], self.Mice[j].hmm.parameters[2][s], self.Mice[i].hmm.parameters[0].T[state], self.Mice[i].hmm.parameters[2][state]))

            kl_divergence = create_kl_divergence_matrix(states_params)
            
            all_kl_matrix = np.log10(np.array(all_kl_divergence) + 1)
            kl_matrix = np.log10(kl_divergence + 1)
            
            annot_array = np.array([[round(item, 2) for item in row] for row in kl_matrix])
            labels = ['$M_{' + str(i+1) + '}$' for i in self.Mice_N]
            fig, axs = plt.subplots(1,1, figsize=(10,8))
            sns.heatmap(kl_matrix, cmap='RdBu', ax = axs, square = 'True', cbar = True, vmin = 0, vmax = 6, annot=annot_array, annot_kws={'size': 14})
            axs.set_title(" (Log-scale) K-L Divergence", fontsize = 25)
            axs.set_xticklabels(labels)
            axs.set_yticklabels(labels, rotation = 0)
            axs.tick_params(axis='both', which='major', labelsize=20)
            plt.tight_layout()
            plt.savefig(self.file_path + self.state_meaning + '_KLMatrix.png')
            
            KL_divergence= Exclude_Diagonal(kl_matrix)
            print('Average Similar State KL: ', np.mean(KL_divergence))
            print('Average All KL: ', np.mean(all_kl_matrix))
            
            statistic, p_value = stats.ks_2samp(KL_divergence, all_kl_divergence)
            fig, axs = plt.subplots(1, 1, figsize=(12, 3))
            axs.hist(KL_divergence, bins = 10, density = True, color = 'red', alpha = 0.5, label = self.state_meaning)
            axs.hist(all_kl_matrix,bins = 10, density = True,  color = 'blue', alpha = 0.5, label = 'Other')
            axs.plot([],[], color = 'white', label = 'p = ' + str(round(p_value, 3)))
            axs.legend(loc = 'upper right', fontsize = 20)
            axs.set_xlabel('(Log-scale) K-L Divergence', fontsize=20)
            axs.set_ylabel('Density', fontsize=20)
            axs.tick_params(axis='both', which='major', labelsize=18)
            axs.spines['top'].set_visible(False)
            axs.spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(self.file_path + self.state_meaning + 'KL_Compare.png')
            if self.print: plt.show()
            plt.close()
        
        def PlotImages(self):
            Speed = []
            Acce = []
            BodyL = []
            
            fig, axs = plt.subplots(1,len(self.Mice_N),figsize = (30, 8))
            for i in range(len(self.Mice_N)):
                state = self.states[self.Mice_N[i]] - 1
                Mouse = self.Mice[self.Mice_N[i]]

                Speed.append(Mouse.hmm.parameters[0].T[state][0])
                Acce.append(Mouse.hmm.parameters[0].T[state][1])
                BodyL.append(Mouse.hmm.parameters[0].T[state][2])
                
                if self.state_meaning == 'WheelMoving':
                    for t in range(30, len(Mouse.arena.pellets.index)):
                        trigger = Mouse.arena.pellets.index[t]
                        start, end = trigger - pd.Timedelta("1S"), trigger
                        trigger_state = Mouse.mouse_pos.loc[start:end, 'state'][0]
                        if trigger_state == state and Mouse.mouse_pos.loc[start:end, 'r'][0] < Mouse.arena.radius:
                            if Mouse.mouse_pos.loc[start:end, 'state'][-1] == state:
                                break
                    id = Mouse.arena.visits['start'].searchsorted(trigger, side='right') - 1
                    patch = Mouse.arena.visits['patch'][id]
                    
                if self.state_meaning == 'Grooming':
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
            plt.savefig(self.file_path + self.state_meaning + '_Frames.png')
            if self.print: plt.show()
            plt.close()

            print('Avegra Speed: ', np.mean(np.array(Speed))*2e-3, 'm/s')
            print('Avegrage Acce: ', np.mean(np.array(Acce))*2e-3, 'm/s2')
            print('Avegra Body Length: ', np.mean(np.array(BodyL))*2, 'mm')
        
        def Display(self, K_L, Images, Parameter, file_path, print = False):
            self.file_path = file_path 
            self.print = print
            
            if K_L: self.PlotKLDivergence()
            if Images: self.PlotImages()
            if Parameter: self.PlotParamters()

    class Event_Sequence:
        def __init__(self, Summary, event_name, left_seconds, right_seconds):
            self.left_seconds = left_seconds
            self.right_seconds = right_seconds
            self.event_name = event_name
            self.KL_M = Summary.KL_M
            
            self.file_path = None
            self.file_name = None
            self.print = False 
            
            self.Mice = Summary.Mice
            self.Mice_N = Summary.Mice_N
        
        def Compare(self):
            SEQUENCE = []
            start, end = int((self.left_seconds-1.5) * 10), int((self.left_seconds+1.5) * 10)
            for i in range(len(self.Mice_N)):
                Mouse = self.Mice[self.Mice_N[i]]
                if self.event_name == 'Pellet Delivery': Events = Mouse.arena.pellets.index
                if self.event_name == 'Start Visit': Events = Mouse.arena.visits.dropna(subset=['speed'])['start']
                if self.event_name == 'End Visit': Events = Mouse.arena.visits.dropna(subset=['speed'])['end']
                if self.event_name == 'Enter Arena': Events = Mouse.arena.entry 
                STATES = Mouse.hmm.process_states.Event_Triggering(Mouse.mouse_pos, Events, self.left_seconds, self.right_seconds, 'state', insert_nan = 0)
                sequence = Mouse.hmm.process_states.Find_Event_Sequence(STATES, penalty = 0)
                SEQUENCE.append(sequence[start:end])
            
            Distance = np.zeros((len(self.Mice_N), len(self.Mice_N)))
            for i in range(len(self.Mice_N)):
                for j in range(len(self.Mice_N)):
                    Distance[i,j] = Compare_Sequence(SEQUENCE[i], SEQUENCE[j], np.log10(self.KL_M[i,j]+1))

            annot_array = np.array([[round(item, 1) for item in row] for row in Distance])
            row_labels = [str(i+1) for i in self.Mice_N ]
            column_labels = [str(i+1) for i in self.Mice_N ]
            
            fig, axs = plt.subplots(1,1,figsize = (20, 12))
            sns.heatmap(Distance, cmap='RdBu', ax = axs, square = 'True', cbar = False, annot=annot_array, annot_kws={'size': 20})
            axs.set_title("Dominant Sequence Comparison", fontsize = 32)
            axs.set_xlabel('Session', fontsize = 24)
            axs.set_ylabel('Session', fontsize = 24)
            axs.set_xticks(np.arange(0.5, len(column_labels)+0.5, 1))
            axs.set_xticklabels(column_labels, rotation = 0)
            axs.set_yticks(np.arange(0.5,len(row_labels)+0.5,1))
            axs.set_yticklabels(row_labels, rotation = 0)
            axs.tick_params(axis='both', which='major', labelsize=24)
            plt.tight_layout()
            plt.savefig(self.file_path + 'Compare_' + self.file_name + '.png')
            
            print('Median KL-Divergence: ', np.median(Exclude_Diagonal(Distance)))
        
        
        def Display(self, file_path, file_name, print = False):
            self.file_path = file_path 
            self.file_name = file_name
            self.print = print
            
            self.Compare()
            
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