import cv2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import scipy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import sys
from pathlib import Path

current_script_path = Path(__file__).resolve()
parent_dir = current_script_path.parent.parent
sys.path.insert(0, str(parent_dir))

import aeon
import aeon.io.api as api
from aeon.io import reader, video
from aeon.schema.dataset import exp02
from aeon.analysis.utils import visits, distancetravelled

root = [Path("/ceph/aeon/aeon/data/raw/AEON2/experiment0.2")]

def assign_marker(dt):
    if dt.hour >= 19:  # If time is 7 PM or later, the segment belongs to the next day
        return (dt + pd.Timedelta(days=1)).date()
    else:  # If time is before 7 PM, the segment belongs to the current day
        return dt.date()
    
def calculate_cr(hour):
    if 8 <= hour < 19:  # From 8 AM to 7 PM
        return 0
    elif 19 <= hour < 20:
        return (hour - 19)
    elif 7 <= hour < 8: 
        return 1 - (hour - 7)
    else:
        return 1

def SeparateDF(df, freq = '3H'):
    groups = df.groupby(pd.Grouper(freq=freq))
    dfs = []
    for _, group in groups:
        if not group.empty: dfs.append(group)
    
    return dfs


def AddWeight(mouse_pos, weight):
    mouse_pos.index = pd.to_datetime(mouse_pos.index)
    weight.index = pd.to_datetime(weight.index)

    mouse_pos.sort_index(inplace=True)
    weight.sort_index(inplace=True)

    # Initialize 'weight' column in mouse_pos
    mouse_pos['weight'] = pd.Series(index=mouse_pos.index)

    # Get the first recorded weight
    first_recorded_weight = weight.iloc[0]['value']

    # Fill the mouse_pos 'weight' column
    for idx in mouse_pos.index:
        # Check if the current index is before the first recorded weight
        if idx < weight.index[0]:
            mouse_pos.at[idx, 'weight'] = first_recorded_weight
        else:
            # Find the most recent weight entry that's less than or equal to the current index
            recent_weight = weight[weight.index <= idx].iloc[-1]['value']
            mouse_pos.at[idx, 'weight'] = recent_weight
            
def PositionInPatch(mouse_pos, r = 30):
    Patch1, Patch2 = [554,832],[590.25, 256.75]
    
    # Calculate distances
    distance1 = np.sqrt((mouse_pos['smoothed_position_x'] - Patch1[0]) ** 2 + (mouse_pos['smoothed_position_y'] - Patch1[1]) ** 2)
    distance2 = np.sqrt((mouse_pos['smoothed_position_x'] - Patch2[0]) ** 2 + (mouse_pos['smoothed_position_y'] - Patch2[1]) ** 2)
    
    # Calssify
    mouse_pos['Patch'] = np.where(distance1 < r, 1, 0) + np.where(distance2 < r, 1, 0)
    
    return mouse_pos

def PositionInArena(mouse_pos):
    X, Y = mouse_pos.x.to_numpy(), mouse_pos.y.to_numpy()
    x_o = (min(X) + max(X))/2
    y_o = (min(Y) + max(Y))/2
    r = ((max(X)-min(X))/2 + (max(Y)-min(Y))/2)/2 - 10
    
    distance = np.sqrt((mouse_pos['smoothed_position_x'] - x_o) ** 2 + (mouse_pos['smoothed_position_y'] - y_o) ** 2)
    mouse_pos['Arena'] = np.where(distance < r, 1, 0)
    
    return mouse_pos
    
        
def InPatch(mouse_pos, r = 30, interval = 5):
    Patch1, Patch2 = [554,832],[590.25, 256.75]
    
    # Calculate distances
    distance1 = np.sqrt((mouse_pos['smoothed_position_x'] - Patch1[0]) ** 2 + (mouse_pos['smoothed_position_y'] - Patch1[1]) ** 2)
    distance2 = np.sqrt((mouse_pos['smoothed_position_x'] - Patch2[0]) ** 2 + (mouse_pos['smoothed_position_y'] - Patch2[1]) ** 2)
    
    # Calssify
    mouse_pos['Patch1'] = np.where(distance1 < r, 1, 0)
    mouse_pos['Patch2'] = np.where(distance2 < r, 1, 0)
    
    # Fix Classification
    groups = mouse_pos['Patch1'].ne(mouse_pos['Patch1'].shift()).cumsum()
    zeros_groups = mouse_pos[mouse_pos['Patch1'] == 0].groupby(groups)['Patch1']
    for name, group in zeros_groups:
        duration = group.index[-1] - group.index[0]
        if duration < pd.Timedelta(seconds=interval): mouse_pos.loc[group.index, 'Patch1'] = 1
        
    groups = mouse_pos['Patch2'].ne(mouse_pos['Patch2'].shift()).cumsum()
    zeros_groups = mouse_pos[mouse_pos['Patch2'] == 0].groupby(groups)['Patch2']
    for name, group in zeros_groups:
        duration = group.index[-1] - group.index[0]
        if duration < pd.Timedelta(seconds=interval): mouse_pos.loc[group.index, 'Patch2'] = 1

        

def MoveWheel(start, end, patch = 'Patch1', interval_seconds = 10):
    if patch == 'Patch1': encoder = api.load(root, exp02.Patch1.Encoder, start=start, end=end)
    else: encoder = api.load(root, exp02.Patch2.Encoder, start=start, end=end)
    
    w = -distancetravelled(encoder.angle).to_numpy()
    dw = np.concatenate((np.array([0]), w[:-1]- w[1:]))
    encoder['Distance'] = pd.Series(w, index=encoder.index)
    encoder['DistanceChange'] = pd.Series(dw, index=encoder.index)
    encoder['DistanceChange'] = encoder['DistanceChange'].rolling('10S').mean()
    encoder['Move'] = np.where(encoder.DistanceChange > 0.001, 1, 0)
    
    if interval_seconds < 0.01: return encoder
    groups = encoder['Move'].ne(encoder['Move'].shift()).cumsum()
    zeros_groups = encoder[encoder['Move'] == 0].groupby(groups)['Move']
    for name, group in zeros_groups:
        duration = group.index[-1] - group.index[0]
        if duration < pd.Timedelta(seconds=interval_seconds): encoder.loc[group.index, 'Move'] = 1
    return encoder


def AtGate(mouse_pos, r=5):
    gate = [1233, 588]
    distance = np.sqrt((mouse_pos['smoothed_position_x'] - gate[0]) ** 2 + (mouse_pos['smoothed_position_y'] - gate[1]) ** 2)

    return mouse_pos.iloc[np.where(distance < r)].index


def Visits(mouse_pos, patch = 'Patch1', pre_period_seconds = 10):
    encoder = MoveWheel(mouse_pos.index[0], mouse_pos.index[-1], patch = patch)
        
    entry = AtGate(mouse_pos)
        
    Visits = {'start':[],'end':[], 'distance':[], 'duration':[], 'speed':[], 'acceleration':[], 'weight':[],'state':[], 'entry':[]}
    
    groups = encoder['Move'].ne(encoder['Move'].shift()).cumsum()
    visits = encoder[encoder['Move'] == 1].groupby(groups)['Move']
    for name, group in visits:
        Visits['start'].append(group.index[0])
        Visits['end'].append(group.index[-1])
        Visits['duration'].append((group.index[-1]-group.index[0]).total_seconds())
        Visits['distance'].append(encoder.loc[group.index[0], 'Distance']-encoder.loc[group.index[-1], 'Distance'])
            
        pre_end = group.index[0]
        pre_start = pre_end - pd.Timedelta(seconds = pre_period_seconds)
        if pre_start < mouse_pos.index[0]: pre_start = mouse_pos.index[0]
            
        pre_visit_data = mouse_pos.loc[pre_start:pre_end]
        
        Visits['speed'].append(pre_visit_data['smoothed_speed'].mean())
        Visits['acceleration'].append(pre_visit_data['smoothed_acceleration'].mean())
        Visits['weight'].append(pre_visit_data['weight'].mean())
        Visits['state'].append(pre_visit_data['states'].value_counts().idxmax())
            

        index = entry.searchsorted(pre_end, side='left') - 1
        index = max(index, 0)
        Visits['entry'].append((pre_end - entry[index]).total_seconds())
    
    return pd.DataFrame(Visits)

def VisitIntervals(Visits_Patch1, Visits_Patch2):
        Patch1, Patch2 = True, True
        if len(Visits_Patch1) == 0: Patch1 = False
        if len(Visits_Patch2) == 0: Patch2 = False
        
        Visits_Patch1 = Visits_Patch1.copy()
        Visits_Patch2 = Visits_Patch2.copy()
        
        Visits_Patch1['PelletsInLastVisitSelf'] = 0
        Visits_Patch1['PelletsInLastVisitOther'] = 0
        Visits_Patch1['IntervalLastVisit'] = 0
        for i in range(1,len(Visits_Patch1)):
                start, end = Visits_Patch1.start[i-1], Visits_Patch1.end[i-1]
                
                pellets_patch1 = api.load(root, exp02.Patch1.DeliverPellet, start=start, end=end)
                Visits_Patch1.loc[i, 'PelletsInLastVisitSelf'] = len(pellets_patch1)
                Visits_Patch1.loc[i, 'IntervalLastVisit'] = (Visits_Patch1.start[i] - end).total_seconds()
                
                if Patch2:
                        prior_timestamps = Visits_Patch2[Visits_Patch2['end'] < Visits_Patch1.start[i]]
                        if not prior_timestamps.empty:
                                start_, end_ = prior_timestamps.iloc[-1]['start'], prior_timestamps.iloc[-1]['end']
                                pellets_patch2 = api.load(root, exp02.Patch2.DeliverPellet, start=start_, end=end_)
                                Visits_Patch1.loc[i, 'PelletsInLastVisitOther'] = len(pellets_patch2)
                                if end < end_: Visits_Patch1.loc[i, 'IntervalLastVisit'] = (Visits_Patch1.start[i] - end_).total_seconds()
        
                
        Visits_Patch2['PelletsInLastVisitSelf'] = 0
        Visits_Patch2['PelletsInLastVisitOther'] = 0
        Visits_Patch2['IntervalLastVisit'] = 0
        for i in range(1,len(Visits_Patch2)):
                start, end = Visits_Patch2.start[i-1], Visits_Patch2.end[i-1]

                pellets_patch2 = api.load(root, exp02.Patch2.DeliverPellet, start=start, end=end)
                Visits_Patch2.loc[i, 'PelletsInLastVisitSelf'] = len(pellets_patch2)
                Visits_Patch2.loc[i, 'IntervalLastVisit'] = (Visits_Patch2.start[i] - end).total_seconds()
                
                if Patch1:
                        prior_timestamps = Visits_Patch1[Visits_Patch1['end'] < Visits_Patch2.start[i]]
                        if not prior_timestamps.empty: 
                                start_, end_ = prior_timestamps.iloc[-1]['start'], prior_timestamps.iloc[-1]['end']
                                pellets_patch1 = api.load(root, exp02.Patch1.DeliverPellet, start=start_, end=end_)
                                Visits_Patch2.loc[i, 'PelletsInLastVisitOther'] = len(pellets_patch1)
                                if end < end_: Visits_Patch2.loc[i, 'IntervalLastVisit'] = (Visits_Patch2.start[i] - end_).total_seconds()

        return Visits_Patch1, Visits_Patch2

def VisitPatchPellets(Visits_Patch1, Visits_Patch2):
        Visits_Patch1 = Visits_Patch1.copy()
        Visits_Patch2 = Visits_Patch2.copy()

        Visits_Patch1['DistanceTravelledinVisit'] = 0
        for i in range(len(Visits_Patch1)):
                start, end = Visits_Patch1.start[i], Visits_Patch1.end[i]
                
                encoder1 = api.load(root, exp02.Patch1.Encoder, start=start, end=end)
                '''                
                if encoder1.empty: Visits_Patch1.loc[i, 'DistanceTravelledinVisit'] = 0
                else:
                        w1 = -distancetravelled(encoder1.angle)
                        Visits_Patch1.loc[i, 'DistanceTravelledinVisit'] = w1[0]-w1[-1]'''
                if not encoder1.empty: 
                        w1 = -distancetravelled(encoder1.angle)
                        Visits_Patch1.loc[i, 'DistanceTravelledinVisit'] = w1[0]-w1[-1]
        
        Visits_Patch1['PelletsInLastVisitSelf'] = 0
        Visits_Patch1['PelletsInLastVisitOther'] = 0
        for i in range(1,len(Visits_Patch1)):
                start, end = Visits_Patch1.start[i-1], Visits_Patch1.end[i-1]
                
                pellets_patch1 = api.load(root, exp02.Patch1.DeliverPellet, start=start, end=end)
                Visits_Patch1.loc[i, 'PelletsInLastVisitSelf'] = len(pellets_patch1)
                
                prior_timestamps = Visits_Patch2[Visits_Patch2['end'] < Visits_Patch1.start[i]]
                '''if prior_timestamps.empty: Visits_Patch1.loc[i, 'PelletsInLastVisitOther'] = 0
                else:'''
                if not prior_timestamps.empty:
                        start_, end_ = prior_timestamps.iloc[-1]['start'], prior_timestamps.iloc[-1]['end']
                        pellets_patch2 = api.load(root, exp02.Patch2.DeliverPellet, start=start_, end=end_)
                        Visits_Patch1.loc[i, 'PelletsInLastVisitOther'] = len(pellets_patch2)
                
        Visits_Patch2['DistanceTravelledinVisit'] = 0
        for i in range(len(Visits_Patch2)):
                start, end = Visits_Patch2.start[i], Visits_Patch2.end[i]
                encoder2 = api.load(root, exp02.Patch2.Encoder, start=start, end=end)
                '''                
                if encoder2.empty: Visits_Patch2.loc[i, 'DistanceTravelledinVisit'] = 0
                else:
                        w2 = -distancetravelled(encoder2.angle)
                        Visits_Patch2.loc[i, 'DistanceTravelledinVisit'] = w2[0]-w2[-1]'''
                if not encoder2.empty:
                        w2 = -distancetravelled(encoder2.angle)
                        Visits_Patch2.loc[i, 'DistanceTravelledinVisit'] = w2[0]-w2[-1]
        
                
        Visits_Patch2['PelletsInLastVisitSelf'] = 0
        Visits_Patch2['PelletsInLastVisitOther'] = 0
        for i in range(1,len(Visits_Patch2)):
                start, end = Visits_Patch2.start[i-1], Visits_Patch2.end[i-1]

                pellets_patch2 = api.load(root, exp02.Patch2.DeliverPellet, start=start, end=end)
                Visits_Patch2.loc[i, 'PelletsInLastVisitSelf'] = len(pellets_patch2)
                
                prior_timestamps = Visits_Patch1[Visits_Patch1['end'] < Visits_Patch2.start[i]]
                '''if prior_timestamps.empty: Visits_Patch2.loc[i, 'PelletsInLastVisitOther'] = 0
                else:
                        start_, end_ = prior_timestamps.iloc[-1]['start'], prior_timestamps.iloc[-1]['end']
                        pellets_patch1 = api.load(root, exp02.Patch1.DeliverPellet, start=start_, end=end_)
                        Visits_Patch2.loc[i, 'PelletsInLastVisitOther'] = len(pellets_patch1)'''
                if not prior_timestamps.empty: 
                        start_, end_ = prior_timestamps.iloc[-1]['start'], prior_timestamps.iloc[-1]['end']
                        pellets_patch1 = api.load(root, exp02.Patch1.DeliverPellet, start=start_, end=end_)
                        Visits_Patch2.loc[i, 'PelletsInLastVisitOther'] = len(pellets_patch1)

        return Visits_Patch1, Visits_Patch2    

def DrawTimeInPatch(mouse_pos, pellets_patch1, pellets_patch2):
    start, end = mouse_pos.index[0], mouse_pos.index[-1]

    fig, axs = plt.subplots(6, 1, figsize=(30, 7), sharex=True)

    mouse_pos['Patch2_Visit_Time_Seconds'].plot(ax = axs[0],color = 'green')
    axs[0].set_xticks([]) 
    axs[0].set_yticks([]) 
    axs[0].set_facecolor('white') 
    axs[0].set_ylabel('P2 Visit', fontsize = 12)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].set_xlim(start, end)
    axs[0].set_yticks([])
    #axs[0].set_ylim(0, 1.1)

    for pellet in pellets_patch2.itertuples():
        forage_time = pellet.Index
        axs[1].axvline(pd.Timestamp(forage_time), color='green', linewidth=1)
    axs[1].set_xticks([]) 
    axs[1].set_yticks([]) 
    axs[1].set_facecolor('white') 
    axs[1].set_ylabel('P2', fontsize = 14)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].set_xlim(start, end)

    for pellet in pellets_patch1.itertuples():
        forage_time = pellet.Index
        axs[2].axvline(pd.Timestamp(forage_time), color='brown', linewidth=1)
    axs[2].set_yticks([])  
    axs[2].set_facecolor('white')  
    axs[2].set_ylabel('P1', fontsize = 14)
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    axs[2].set_xlim(start, end)
    
    mouse_pos['Patch1_Visit_Time_Seconds'].plot(ax = axs[3],color = 'brown')
    axs[3].set_xticks([]) 
    axs[3].set_yticks([]) 
    axs[3].set_facecolor('white') 
    axs[3].set_ylabel('P1 Visit', fontsize = 12)
    axs[3].spines['top'].set_visible(False)
    axs[3].spines['right'].set_visible(False)
    axs[3].set_xlim(start, end)
    axs[3].set_yticks([])
    #axs[3].set_ylim(0, 1.1)
    
    mouse_pos['smoothed_speed'].plot(ax = axs[4],color = 'red')
    axs[4].set_xticks([]) 
    axs[4].set_yticks([]) 
    axs[4].set_facecolor('white') 
    axs[4].set_ylabel('Speed', fontsize = 12)
    axs[4].spines['top'].set_visible(False)
    axs[4].spines['right'].set_visible(False)
    axs[4].set_xlim(start, end)
    axs[4].set_yticks([])
    
    mouse_pos['smoothed_acceleration'].plot(ax = axs[5],color = 'blue')
    axs[5].set_xticks([]) 
    axs[5].set_yticks([]) 
    axs[5].set_facecolor('white') 
    axs[5].set_ylabel('Acce.', fontsize = 12)
    axs[5].spines['top'].set_visible(False)
    axs[5].spines['right'].set_visible(False)
    axs[5].set_xlim(start, end)
    axs[5].set_yticks([])
    
    
    plt.show()
    
def DrawVisitInPatch(mouse_pos, pellets_patch1, pellets_patch2):
    start, end = mouse_pos.index[0], mouse_pos.index[-1]

    
    fig, axs = plt.subplots(4, 1, figsize=(30, 5), sharex=True)

    mouse_pos['Patch2'].plot(ax = axs[0],color = 'green')
    axs[0].set_xticks([]) 
    axs[0].set_yticks([]) 
    axs[0].set_facecolor('white') 
    axs[0].set_ylabel('P2 Visit', fontsize = 12)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].set_xlim(start, end)
    axs[0].set_yticks([])
    axs[0].set_ylim(0, 1.1)

    for pellet in pellets_patch2.itertuples():
        forage_time = pellet.Index
        axs[1].axvline(pd.Timestamp(forage_time), color='green', linewidth=1)
    axs[1].set_xticks([]) 
    axs[1].set_yticks([]) 
    axs[1].set_facecolor('white') 
    axs[1].set_ylabel('P2', fontsize = 14)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].set_xlim(start, end)

    for pellet in pellets_patch1.itertuples():
        forage_time = pellet.Index
        axs[2].axvline(pd.Timestamp(forage_time), color='brown', linewidth=1)
    axs[2].set_yticks([])  
    axs[2].set_facecolor('white')  
    axs[2].set_ylabel('P1', fontsize = 14)
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    axs[2].set_xlim(start, end)
    
    mouse_pos['Patch1'].plot(ax = axs[3],color = 'brown')
    axs[3].set_xticks([]) 
    axs[3].set_yticks([]) 
    axs[3].set_facecolor('white') 
    axs[3].set_ylabel('P1 Visit', fontsize = 12)
    axs[3].spines['top'].set_visible(False)
    axs[3].spines['right'].set_visible(False)
    axs[3].set_xlim(start, end)
    axs[3].set_yticks([])
    axs[3].set_ylim(0, 1.1)

    
    plt.show()
    
    
def SimpleLinearRegression(X, y):
    model = LinearRegression()
    model.fit(X, y)

    coefficients = model.coef_
    intercept = model.intercept_

    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    return coefficients, intercept, y_pred, mse, r2

def ComparePrediction(y1, y1_predicted, y2, y2_predicted):  
    fig, axs = plt.subplots(4, 1, figsize=(30, 5), sharex=True)

    axs[0].plot(y2, color = 'green')
    axs[0].set_xticks([]) 
    #axs[0].set_yticks([]) 
    axs[0].set_facecolor('white') 
    axs[0].set_ylabel('P2', fontsize = 12)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)

    axs[1].plot(y2_predicted, color = 'green')
    axs[1].set_xticks([]) 
    #axs[1].set_yticks([]) 
    axs[1].set_facecolor('white') 
    axs[1].set_ylabel('P2 Pred.', fontsize = 14)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)

    axs[2].plot(y1, color = 'brown')
    #axs[2].set_yticks([])  
    axs[2].set_facecolor('white')  
    axs[2].set_ylabel('P1', fontsize = 14)
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    #axs[2].set_xlim(start, end)
    
    axs[3].plot(y1_predicted, color = 'brown')
    axs[3].set_xticks([]) 
    #axs[3].set_yticks([]) 
    axs[3].set_facecolor('white') 
    axs[3].set_ylabel('P1 Pred.', fontsize = 12)
    axs[3].spines['top'].set_visible(False)
    axs[3].spines['right'].set_visible(False)
    #axs[3].set_xlim(start, end)
    
    plt.show()
    
    
    
