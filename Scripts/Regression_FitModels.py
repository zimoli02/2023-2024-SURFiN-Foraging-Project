import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

import sys
import os
from pathlib import Path

aeon_mecha_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(aeon_mecha_dir))
#print(sys.path)

import aeon
import aeon.io.api as api
from aeon.schema.dataset import exp02
from aeon.analysis.utils import visits

import Functions.patch as patch

root = [Path("/ceph/aeon/aeon/data/raw/AEON2/experiment0.2")]

subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])
short_sessions = sessions.iloc[[4,16,17,20,23,24,25,26,28,29,30,31]]
long_sessions = sessions.iloc[[8, 10, 11, 14]]

def ConcatenateSessions():
    dfs = []
    for session, i in zip(list(short_sessions.itertuples()), range(len(short_sessions))):
        title = 'ShortSession'+str(i)
        print(title)
        Visits_Patch1 = pd.read_parquet('../Data/RegressionPatchVisits/' + title + 'Visit1.parquet', engine='pyarrow')
        Visits_Patch2 = pd.read_parquet('../Data/RegressionPatchVisits/' + title + 'Visit2.parquet', engine='pyarrow')
        dfs.append(Visits_Patch1)
        dfs.append(Visits_Patch2)
        
    VISIT = pd.concat(dfs, ignore_index=False)
    VISIT = VISIT[VISIT['distance'] >= 0.1]
    VISIT['interc'] = 1
    
    return VISIT

def Variables(VISIT, feature, predictor = 'duration'):
    X = VISIT[feature]
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), index = X.index, columns = X.columns)
    X['interc'] = 1
    Y = VISIT[predictor]
    
    return X, Y

def Model(X, Y, type = 'Poisson'):
    if type == 'Poisson': model = sm.GLM(Y, X, family=sm.families.Poisson())
    if type == 'Gaussian': model = sm.GLM(Y, X, family=sm.families.Gaussian())
    if type == 'Gamma': model = sm.GLM(Y, X, family=sm.families.Gamma(sm.families.links.Log()))
    
    result = model.fit()
    y_pred = result.predict(X)
    
    return result, y_pred

def main():
    TYPES = ['Poisson', 'Gaussian', 'Gamma']
    
    VISIT = ConcatenateSessions()
    X, Y = Variables(VISIT, feature = ['speed','acceleration', 'weight','PelletsInLastVisitSelf', 'PelletsInLastVisitOther', 'IntervalLastVisit' ,'entry'], predictor='duration')
    
    for TYPE in TYPES:
        result, y_pred = Model(X, Y, type = TYPE)
        
        fig, axs = plt.subplots(2, 1, figsize=(15, 5), sharex=True)

        axs[0].plot(Y.to_numpy(), color = 'black')
        axs[0].set_xticks([]) 
        axs[0].set_facecolor('white') 
        axs[0].set_ylabel('Duration', fontsize = 12)
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)

        axs[1].plot(y_pred.to_numpy(), color = 'blue', label = 'Correlation: '+str(round(np.corrcoef(Y, y_pred)[0,1],4)))
        axs[1].set_xticks([]) 
        axs[1].set_facecolor('white') 
        axs[1].set_ylabel('Dur. Pred.', fontsize = 14)
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[1].legend()
        
        plt.savefig('../Images/Regression/AllSessionsData/' + TYPE + '.png')
        plt.show()
        
        
        fig, axs = plt.subplots(figsize=(20, 8))
        axs.axis('off')
        axs.text(0.5, 0.5, str(result.summary()),
                verticalalignment='center', horizontalalignment='left',
                transform=axs.transAxes, fontsize=12)
        plt.savefig('../Images/Regression/AllSessionsParams/' + TYPE + '.png')
        plt.show()
        

if __name__ == "__main__":
    main()