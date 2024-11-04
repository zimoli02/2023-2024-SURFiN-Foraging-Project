import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import pickle

import statsmodels.api as sm
import scipy.stats as stats
import random
from scipy.special import factorial
from dtaidistance import dtw

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

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

import warnings
import re

# Suppress the scipy warning
warnings.filterwarnings("ignore", message="scipy._lib.messagestream.MessageStream size changed, may indicate binary incompatibility. Expected 56 from C header, got 64 from PyObject")

# Suppress the UserWarning about out-of-order timestamps
warnings.filterwarnings("ignore", message="data index for .* contains out-of-order timestamps!")

# If you want to suppress all UserWarnings from a specific file
warnings.filterwarnings("ignore", category=UserWarning, module="aeon.io.api")

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


'''-------------------------------LDS-------------------------------'''

def Display_Kinematics(Mouse, Trace, Property, file_path):
    Kinematics = result.Kinematics(Mouse)
    kinematics = Kinematics.kinematics(Kinematics)
    kinematics.Display(Trace, Property, file_path)
    
    print('Display_Kinematics Completed')

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
            np.save('../../SocialData/HMMStates/Loglikelihood_' + Mouse_title + '.npy', Loglikelihood)
            print('End Inference for n = ', str(n))
        Loglikelihood = np.array(Loglikelihood)

    points = 24*60*60*10 
    Loglikelihood = Loglikelihood[:,-1]/points
    df = Loglikelihood[1:] - Loglikelihood[:-1]
    
    fig, axs = plt.subplots(1,2,figsize = (20,8))
    axs[0].scatter(N, Loglikelihood)
    axs[0].plot(N, Loglikelihood)
    axs[0].set_xticks(np.arange(N[0], N[-1] + 1, 5))
    
    axs[1].scatter(N[1:], df)
    axs[1].plot(N[1:], df)
    axs[1].set_xticks(np.arange(N[0], N[-1] + 1, 5))
    
    for i in range(2):
        #axs[i].axvline(x=10, color = 'red', linestyle = "--")
        axs[i].set_xlabel('State Number', fontsize = 30)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].tick_params(axis='both', which='major', labelsize=20)
    axs[0].set_ylabel('Log Likelihood per Point', fontsize = 30)
    axs[1].set_ylabel('$\Delta$Log Likelihood per Point', fontsize = 30)
    plt.tight_layout()
    plt.savefig(file_path + Mouse_title + '_New.png')
    plt.close()

def Display_HMM_Model_Feature(HMM, Parameter, Parameter_zoom, TransM, ConnecM, K_L, file_path):
    Model_Feature = HMM.Model_Features(HMM)
    Model_Feature.Display(Parameter, Parameter_zoom, TransM, ConnecM, K_L, file_path)
    print('Display_HMM_Model_Feature Completed')

def Display_HMM_States_Features(HMM, Position, Duration, Frequency, Example, file_path):
    State_Feature = HMM.States_Features(HMM, file_path)
    State_Feature.Display(Position, Frequency, Duration, Example)
    print('Display_HMM_States_Feature Completed')

def Display_HMM_Event_Characterization(HMM, pellet_delivery, start_visit, end_visit, enter_arena, file_path = '../Images/Social_HMM/'):
    Heatmap = True 
    Position = False 
    SequenceDivergence = True
    SequenceDistance = True 
    
    if pellet_delivery: 
        Event_Characterisation = HMM.Characterize_Timepoints(HMM, event_name = 'Pellet Delivery', Events = HMM.Pellets, left_seconds = 3, right_seconds = 3)
        Event_Characterisation.Display(Heatmap, Position, SequenceDivergence, SequenceDistance, file_path = file_path, file_name = 'PelletDelivery')
    if start_visit: 
        Event_Characterisation = HMM.Characterize_Timepoints(HMM, event_name = 'Start Visit', Events = HMM.Starts, left_seconds = 5, right_seconds = 3)
        Event_Characterisation.Display(Heatmap, Position, SequenceDivergence, SequenceDistance, file_path = file_path, file_name = 'EnterVisit')
    if end_visit: 
        Event_Characterisation = HMM.Characterize_Timepoints(HMM, event_name = 'End Visit', Events = HMM.Ends, left_seconds = 3, right_seconds = 5)
        Event_Characterisation.Display(Heatmap, Position, SequenceDivergence, SequenceDistance, file_path = file_path, file_name = 'EndVisit')
    if enter_arena: 
        Event_Characterisation = HMM.Characterize_Timepoints(HMM, event_name = 'Enter Arena', Events = HMM.Entry, left_seconds = 3, right_seconds = 3)
        Event_Characterisation.Display(Heatmap, Position, SequenceDivergence, SequenceDistance, file_path = file_path, file_name = 'EnterArena')
    
    print('Display_HMM_Event_Characterization Completed')


'''-------------------------------PREDICTION-------------------------------'''  

def Display_HMM_States_Predicting_Behavior_Gaussian(Mouse, pellet_delivery = True, start_visit = True, end_visit = True, enter_arena = True, file_path = '../Images/Social_HMM_Prediction/'):

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
        plt.savefig(file_path + file_name + '/' + Mouse_title + '_Prediction.png')
        
        fig, axs = plt.subplots(1, 1, figsize=(50, 4))
        mouse_pos_ = mouse_pos[Predicting_Chunk_start:]
        mouse_pos_.prob.plot(ax = axs, color = 'black')
        for i in range(len(Events)):
            trigger = Events[i]
            axs.axvline(x = trigger, color = 'red')
        plt.savefig(file_path + file_name + '/' + Mouse_title + '_Prediction_Full.png')
        
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

def Display_HMM_States_Predicting_Behavior_Intensity(Mouse, model = 'Poisson', pellet_delivery = True, start_visit = True, end_visit = True, enter_arena = True, file_path = '../Images/Social_HMM_Prediction/'):
    
    def Validate(regression, prediction, evaluator, file_name):            
        evaluators = []
        for i in range(1000):
            permutation = np.random.permutation(len(regression.X))
            regression.X = regression.X.iloc[permutation].set_index(regression.X.index)
            
            if model == 'Gaussian':
                model_ = sm.GLM(regression.Y, regression.X, family=sm.families.Gaussian())
            if model == 'Poisson':
                model_ = sm.GLM(regression.Y, regression.X, family=sm.families.Poisson(sm.families.links.Log()))

            result = model_.fit()
            y_pred = result.predict(prediction.X)
            y, y_pred = prediction.Y.to_numpy().reshape(1,-1)[0], y_pred.to_numpy()
                
            epsilon = 1e-10
            y_pred = np.where(y_pred <= 0, epsilon, y_pred)
            
            if model == 'Poisson': evaluators.append(np.mean(y * np.log(y_pred) - y_pred - np.log(factorial(y))))
            if model == 'Gaussian': evaluators.append(np.corrcoef(y, y_pred  )[0,1])

        print('True evaluator: ', evaluator[1])
        print('Evaluator by chance: ', np.mean(evaluators))
        np.save(file_name + '_' + Mouse_title + '.npy', evaluators)
            
        t_stat, p_value = stats.ttest_1samp(np.array(evaluators), evaluator[1])
        fig, axs = plt.subplots(1, 1, figsize=(6, 8))
        axs.hist(evaluators, bins = 20, color = 'blue', alpha = 0.8)
        axs.axvline(x = evaluator[1], color = 'red', label = 'p-value: ' + str(round(p_value,3)))
        axs.legend(fontsize = 16, loc = 'upper left')
        axs.tick_params(axis='both', which='major', labelsize=14)
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(file_path + file_name + '/' + model + '_Validation_' + Mouse_title + '.png')
        plt.savefig(file_path + file_name + '/' + model + '_Validation_' + Mouse_title + '.eps', format='eps')
        plt.close()
        
        print('Plot Validation')
        
    def Plot(mouse_pos_test, evaluator, file_name):
        
        fig, axs = plt.subplots(1, 1, figsize=(50, 8), sharex=True)
        mouse_pos_test.intensity.plot(ax = axs, color = 'red', linewidth = 5, label = 'Observed')
        axs.set_ylabel('$\lambda(t)$', fontsize = 30)
        axs.legend(fontsize=20, loc = 'upper left')
        axs.tick_params(axis='both', which='major', labelsize=16)
        axs.spines['top'].set_visible(False)
        x_min, x_max = axs.get_xlim()
        axs.set_xlim(left=x_min, right=x_max)
        
        axs_ = axs.twinx()
        mouse_pos_test.intensity_pred.plot(ax = axs_, color = 'black', label = 'Predicted')
        axs_.plot([], [], color = 'white', label = evaluator[0] + '= ' + str(round(evaluator[1],3)))
        axs_.set_ylabel('$\hat{\lambda}(t)$', fontsize = 30)
        axs_.legend(fontsize=20, loc = 'upper right')
        axs_.tick_params(axis='both', which='major', labelsize=16)
        axs_.spines['top'].set_visible(False)
        axs_.set_xlim(left=x_min, right=x_max)
        
        plt.tight_layout()
        plt.savefig(file_path + file_name + '/' + model + '_' + Mouse_title + '.png')
        plt.savefig(file_path + file_name + '/' + Mouse_title + '_' + model + '.eps', format='eps')
        plt.close()
        
        print('Plot Prediction')
        
    def Event_Intensity(mouse_pos_, Events):
        event_timestamps = Events

        start_time = mouse_pos_.index[0]
        event_seconds = (event_timestamps - start_time).total_seconds().values

        event_intensities = np.ones_like(event_seconds)
        
        max_seconds = (mouse_pos_.index[-1]- start_time).total_seconds()
        prediction_seconds = np.arange(0, max_seconds + 1, 1)
        prediction_timestamps = start_time + pd.to_timedelta(prediction_seconds, unit='s')

        kde = stats.gaussian_kde(event_seconds, weights=event_intensities, bw_method=0.001)
        smoothed_intensities = kde(prediction_seconds)

        smoothed_intensities = smoothed_intensities / np.max(smoothed_intensities)
        
        mouse_pos_['intensity'] = 0
        nearest_indices = mouse_pos_.index.get_indexer(prediction_timestamps, method='nearest')
        for i in range(len(nearest_indices)):
            mouse_pos_.iloc[nearest_indices[i], mouse_pos_.columns.get_loc('intensity')] = smoothed_intensities[i]

        print('Calculate Intensity')
        
        return mouse_pos_
    
    def Train(mouse_pos_train):
        regression = mouse.Regression(mouse_pos_train)
        regression.predictor = 'intensity'
        regression.regressor = ['State' + str(i) for i in range(N)] + ['State' + str(i) + '_2' for i in range(N)]
        
        if model == 'Poisson': result = regression.Poisson()
        if model == 'Logistic': result = regression.Logistic()
        if model == 'Gaussian': result = regression.Linear_Regression()
        
        return regression, result
        
    def Test(mouse_pos_test, regression, result):
        prediction = mouse.Regression(mouse_pos_test)
        prediction.predictor = 'intensity'
        prediction.regressor = regression.regressor 
        prediction.Get_Variables()
        
        y_pred = result.predict(prediction.X)
        y, y_pred = prediction.Y.to_numpy().reshape(1,-1)[0], y_pred.to_numpy() 
        
        epsilon = 1e-10
        y_pred = np.where(y_pred <= 0, epsilon, y_pred)
        
        if model == 'Poisson': evaluator = ('Ave. Log Likelihood', np.mean(y * np.log(y_pred) - y_pred - np.log(factorial(y))))
        if model == 'Gaussian': evaluator = ('Correlation', np.corrcoef(y, y_pred)[0,1])
        
        return prediction, y, y_pred, evaluator   
    
    def Predict(event_name, Events, file_name):
        mouse_pos_ = mouse_pos.copy()[::100] # dt = 10s
        print(mouse_pos_.index[1] - mouse_pos_.index[0])
        
        mouse_pos_ = Event_Intensity(mouse_pos_, Events)
    
        mouse_pos_train = mouse_pos_[Active_Chunk[0]:Active_Chunk[1]]
        mouse_pos_test = mouse_pos_[Predicting_Chunk[0]:Predicting_Chunk[1]]
        
        regression, result = Train(mouse_pos_train)
        #print(result.summary())
        print('model training')

        prediction, y, y_pred, evaluator = Test(mouse_pos_test, regression, result)
        print('prediction')

        mouse_pos_test.loc[mouse_pos_test.index, 'intensity_pred'] = y_pred
        Plot(mouse_pos_test, evaluator, file_name)
        
        Validate(regression, prediction, evaluator, file_name)
        
        print('Predicton for ' + event_name + ' Completed') 
    
    Mouse_title = Mouse.type + '_' + Mouse.mouse
    N = Mouse.hmm.n_state
    mouse_pos = Mouse.mouse_pos
    mouse_pos['state'] = pd.Series(Mouse.hmm.states, index = mouse_pos.index)

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
    
    
    if pellet_delivery: Predict(event_name = 'Pellet Delivery', Events = Pellets, file_name = 'PelletDelivery')
    if start_visit: Predict(event_name = 'Move Wheel', Events = pd.DatetimeIndex(Starts), file_name = 'StartVisit')
    if end_visit: Predict(event_name = 'Leave Wheel', Events = pd.DatetimeIndex(Ends), file_name = 'EndVisit')
    if enter_arena: Predict(event_name = 'Enter Arena', Events = Entry, file_name = 'EnterArena')

    print('Display_HMM_States_Predicting_Behavior_Poisson Completed')

def Display_HMM_States_Predicting_Behavior_Logistic(Mouse, pellet_delivery = True, start_visit = True, end_visit = True, enter_arena = True, file_path = '../Images/Social_HMM_Prediction/'):

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
        plt.savefig(file_path + file_name + '/' + Mouse_title + '_Logist.png')
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

'''-------------------------------FORAGING BEHAVIOUR-------------------------------'''  
def Display_Gate_to_Patch(Behaviour, Position, Duration, Distance, file_path):
    Behaviour_Event = Behaviour.Gate_to_Patch(Behaviour)
    Behaviour_Event.Display(Position, Duration, Distance, file_path)
    

def main():
    
    print('Start Loading')
    with open('../Report/Data/Mice.pkl', 'rb') as file:
        Mice = pickle.load(file)
    print('Complete Loading')

    for i in range(len(LABELS)):
        aeon_exp_name, type_name, mouse_name = LABELS[i][0], LABELS[i][1], LABELS[i][2]
        print('Start Processing: ', type_name, "-", mouse_name)
        Mouse = Mice[i]
        
        ''' Mouse = mouse.Mouse(aeon_exp = aeon_exp_name, type = type_name, mouse = mouse_name)
        Mouse.Run_Visits()'''
    
        '''-------------------------------LDS-------------------------------'''

        #Display_Kinematics(Mouse, Trace = True, Property = True, file_path = '../Images/Kinematics/')
    
        '''-------------------------------HMM-------------------------------'''
        #Display_Model_Selection(Mouse, N = np.arange(3, 50), file_path = '../Images/Social_HMM/StateNumber/')

        #Mouse.hmm.Fit_Model(n_state = 10, feature = 'Kinematics_and_Body')
        Mouse.hmm.Get_States(n_state = 10, feature = 'Kinematics_and_Body')
        '''
        HMM = result.HMM(Mouse)

        Display_HMM_Model_Feature(HMM, Parameter = True, Parameter_zoom = True, TransM = True, K_L = True, ConnecM = True, file_path = '../Images/HMM/Model_Feature/')
        Display_HMM_States_Features(HMM, Position = True, Duration = True, Frequency = True, Example = True, file_path = '../Images/HMM/States_Feature/')
        Display_HMM_Event_Characterization(HMM, 
                                            pellet_delivery = True,
                                            start_visit = True,
                                            end_visit = True,
                                            enter_arena = True,
                                            file_path = '../Images/HMM/Event/')'''
        
        
        '''-------------------------------PREDICTION-------------------------------'''  
        
        '''
        Display_HMM_States_Predicting_Behavior_Gaussian(Mouse,
                                                        pellet_delivery = True,
                                                        start_visit = True,
                                                        end_visit = True,
                                                        enter_arena = True,
                                                        file_path = '../Images/Social_HMM_Prediction/')
        
        
        Display_HMM_States_Predicting_Behavior_Logistic(Mouse,
                                                    pellet_delivery = True,
                                                    start_visit = True,
                                                    end_visit = True,
                                                    enter_arena = True,
                                                    file_path = '../Images/Social_HMM_Prediction/')
        '''
        Display_HMM_States_Predicting_Behavior_Intensity(Mouse,
                                                        model = 'Gaussian',
                                                        pellet_delivery = True,
                                                        start_visit = True,
                                                        end_visit = True,
                                                        enter_arena = False,
                                                        file_path = '../Images/HMM_Prediction/')
        '''
        Display_HMM_States_Predicting_Behavior_Intensity(Mouse,
                                                        model = 'Gaussian',
                                                        pellet_delivery = True,
                                                        start_visit = True,
                                                        end_visit = True,
                                                        enter_arena = True,
                                                        file_path = '../Images/Social_HMM_Prediction/')
        
        
        Display_HMM_States_Predicting_Behavior_SeqDetect(Mouse,
                                                    pellet_delivery = True,
                                                    start_visit = True,
                                                    end_visit = True,
                                                    enter_arena = True,
                                                    file_path = '../Images/Social_HMM_Prediction/')
                                                    '''
        
        '''-------------------------------REGRESSION-------------------------------'''                                          

        '''
        Display_Visit_Prediction(Mouse.arena.visits, model = 'linear', file_path = '../Images/Social_Regression/'+Mouse.type+'-'+Mouse.mouse+'/', title = 'Linear_Regression.png')                                            
        Display_Visit_Prediction(Mouse.arena.visits, model = 'MLP', file_path = '../Images/Social_Regression/'+Mouse.type+'-'+Mouse.mouse+'/', title = 'MLP.png')
        '''
        
        '''-------------------------------FORAGING-------------------------------'''  
        '''
        Behaviour = result.Behaviour(Mouse)
        Display_Gate_to_Patch(Behaviour, Position = True, Duration = True, Distance = True, file_path = '../Images/Behaviour/GateToPatch/')
        Display_Learning_Pellets(Mouse, left_seconds=3, right_seconds = 3, file_path='../Images/LearningPellets/')
        '''
        

if __name__ == "__main__":
        main()
        
        
        
        