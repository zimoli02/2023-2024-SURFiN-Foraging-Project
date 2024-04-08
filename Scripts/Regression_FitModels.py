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
short_sessions = sessions.iloc[[4,16,17,20,23,24,25,28,29,30,31]]
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

def Variables(VISIT, feature, predictor = 'distance'):
    X = VISIT[feature]
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), index = X.index, columns = X.columns)
    X['interc'] = 1
    Y = VISIT[predictor]
    #Y  = Y // 0.04
    
    return X, Y

def FitModel(X, Y, type):
    if type == 'Poisson': model = sm.GLM(Y, X, family=sm.families.Poisson())
    if type == 'Gaussian': model = sm.GLM(Y, X, family=sm.families.Gaussian())
    if type == 'Gamma': model = sm.GLM(Y, X, family=sm.families.Gamma(sm.families.links.Log()))
    return model

def CrossValidation(X, Y, type, split_perc = 0.75):
    split_size = int(len(Y) * split_perc)
    indices = np.arange(len(Y))
    
    CORR = []
    corr_max = -1
    
    for i in range(1000):
        np.random.shuffle(indices)
        
        train_indices = indices[:split_size]
        test_indices = indices[split_size:]

        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        Y_train, Y_test = Y.iloc[train_indices], Y.iloc[test_indices]
        
        model = FitModel(X_train,Y_train,type)
        result = model.fit()
        
        Y_test_pred = result.predict(X_test)
        corr = np.corrcoef(Y_test, Y_test_pred)[0,1]
        CORR.append(corr)
        
        if corr > corr_max: model_valid = model
    
    return model_valid, np.mean(CORR)
        
    
def Model(X, Y, type = 'Poisson'):
    model, average_corre = CrossValidation(X, Y, type)
    
    result = model.fit()
    y_pred = result.predict(X)
    
    return result, y_pred, average_corre

def PlotModelPrediction(obs, pred, predictor = 'Distance', TYPE = 'Poisson'):
    fig, axs = plt.subplots(2, 1, figsize=(15, 5), sharex=True)
    axs[0].plot(obs.to_numpy(), color = 'black')
    axs[0].set_xticks([]) 
    axs[0].set_facecolor('white') 
    axs[0].set_ylabel(predictor, fontsize = 12)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)

    axs[1].plot(pred.to_numpy(), color = 'blue', label = 'Correlation: '+str(round(np.corrcoef(obs, pred)[0,1],4)))
    axs[1].set_xticks([]) 
    axs[1].set_facecolor('white') 
    axs[1].set_ylabel(predictor + ' Pred.', fontsize = 14)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].legend()
        
    plt.savefig('../Images/Regression/AllSessionsData/' + TYPE + '.png')
    plt.show()

def PlotModelPrediction_Scatter(obs, pred, predictor = 'Distance', TYPE = 'Poisson'):
    fig, axs = plt.subplots(1, 1, figsize=(10, 10), sharex=True)
    x = obs.to_numpy()
    y = pred.to_numpy()
    axs.scatter(x, y, s = 2)
    
    axs.set_aspect('equal', adjustable='box')
    axs.set_xticks([]) 
    axs.set_ylabel(predictor, fontsize = 12)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('../Images/Regression/AllSessionsData_Scatter/' + TYPE + '.png')
    plt.show()

def PrintModelSummary(result, TYPE):
    fig, axs = plt.subplots(figsize=(20, 8))
    axs.axis('off')
    axs.text(0.5, 0.5, str(result.summary()),
                verticalalignment='center', horizontalalignment='left',
                transform=axs.transAxes, fontsize=12)
    plt.savefig('../Images/Regression/AllSessionsParams/' + TYPE + '.png')
    plt.show()   


def main():
    TYPES = ['Poisson', 'Gaussian', 'Gamma']
    PREDICTOR = 'distance'
    
    VISIT = ConcatenateSessions()
    X, Y = Variables(VISIT, feature = ['speed_x','speed_y','acceleration_x', 'acceleration_y', 'PelletsInLastVisitSelf', 'PelletsInLastVisitOther', 'IntervalLastVisit' ,'entry'], predictor=PREDICTOR)
    
    for TYPE in TYPES:
        result, y_pred, average_corre = Model(X, Y, type = TYPE)
        print("Average Correlation for " + TYPE + " Model Fitted: ", average_corre)
        
        PlotModelPrediction(Y, y_pred, predictor=PREDICTOR, TYPE = TYPE)
        PlotModelPrediction_Scatter(Y, y_pred, predictor=PREDICTOR, TYPE = TYPE)
        
        #PrintModelSummary(result, TYPE)
        
        
        
        

if __name__ == "__main__":
    main()