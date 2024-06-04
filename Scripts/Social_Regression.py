import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from collections import Counter
from sklearn.preprocessing import PolynomialFeatures

import sys
from pathlib import Path

current_script_path = Path(__file__).resolve()
function_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(function_dir))
import Functions.patch as patch

parent_dir = current_script_path.parents[2] / 'aeon_mecha' 
sys.path.insert(0, str(parent_dir))
import aeon
import aeon.io.api as api
from aeon.schema.dataset import exp02
from aeon.analysis.utils import visits
from aeon.schema.schemas import social02

nodes_name = ['nose', 'head', 'right_ear', 'left_ear', 'spine1', 'spine2','spine3', 'spine4']

INFO = pd.read_parquet('../SocialData/INFO3.parquet', engine='pyarrow')
TYPE = ['Pre','Post']
MOUSE = ['BAA-1104045', 'BAA-1104047']
LABELS = [
    ['Pre','BAA-1104045'],
    ['Pre','BAA-1104047'],
    ['Post','BAA-1104045'],
    ['Post','BAA-1104047']
]


def ConcatenateSessions():
    dfs = []
    for i in range(len(LABELS)):
        type, mouse = LABELS[i][0], LABELS[i][1]
        Visits = pd.read_parquet('../SocialData/VisitData/'  + type + "_" + mouse +'_Visit.parquet', engine='pyarrow')
        
        Visits = Visits.dropna(subset=['speed'])
        Visits['distance'] = abs(Visits['distance'])
        Visits = Visits[Visits['distance'] >= 0.1]
        
        dfs.append(Visits)
        
    VISIT = pd.concat(dfs, ignore_index=True)
    
    return VISIT

def Process_Visits():
    for i in range(len(LABELS)):
        print(i)
        type, mouse = LABELS[i][0], LABELS[i][1]
        mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
        
        try:
            Visits_Patch1 = pd.read_parquet('../SocialData/VisitData/'  + type + "_" + mouse +'_Visit1.parquet', engine='pyarrow')
        except FileNotFoundError:
            Visits_Patch1 = patch.Social_Visits(mouse_pos, patch = 'Patch1', pre_period_seconds = 30, arena_r = 511)
            Visits_Patch1.to_parquet('../SocialData/VisitData/'  + type + "_" + mouse +'_Visit1.parquet', engine='pyarrow')
        
        try:
            Visits_Patch2 = pd.read_parquet('../SocialData/VisitData/'  + type + "_" + mouse +'_Visit2.parquet', engine='pyarrow')
        except FileNotFoundError:
            Visits_Patch2 = patch.Social_Visits(mouse_pos, patch = 'Patch2', pre_period_seconds = 30, arena_r = 511)
            Visits_Patch2.to_parquet('../SocialData/VisitData/'  + type + "_" + mouse +'_Visit2.parquet', engine='pyarrow')
        
        try:
            Visits_Patch3 = pd.read_parquet('../SocialData/VisitData/'  + type + "_" + mouse +'_Visit3.parquet', engine='pyarrow')
        except FileNotFoundError:
            Visits_Patch3 = patch.Social_Visits(mouse_pos, patch = 'Patch3', pre_period_seconds = 30, arena_r = 511)
            Visits_Patch3.to_parquet('../SocialData/VisitData/'  + type + "_" + mouse +'_Visit3.parquet', engine='pyarrow')
        
        Visits = patch.VisitIntervals([Visits_Patch1, Visits_Patch2, Visits_Patch3]) 
        Visits.to_parquet('../SocialData/VisitData/'  + type + "_" + mouse +'_Visit.parquet', engine='pyarrow')


def Variables(VISIT, feature, predictor = 'distance'):
    X = VISIT[feature]
    #scaler = StandardScaler()
    # X = pd.DataFrame(scaler.fit_transform(X), index = X.index, columns = X.columns)
    #X['interc'] = 1
    Y = VISIT[predictor]
    
    ''' 
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)  
    
    feature_names = poly.get_feature_names_out(input_features=['last_pellets_self', 'last_pellets_other', 'interval', 'last_interval', 'last_pellets_interval', 'entry'])
    
    X_poly_sm = sm.add_constant(X_poly)  # Add constant term for intercept
    
    
    model = sm.OLS(Y, X_poly_sm).fit()
    coef_df = pd.DataFrame({
        'Feature': list(feature_names),
        'Coefficient': model.params
    })

    print(coef_df)'''
    
    return X, Y

def FitModel(X, Y, type):
    if type == 'Poisson': model = sm.GLM(Y, X, family=sm.families.Poisson())
    if type == 'Gaussian': model = sm.GLM(Y, X, family=sm.families.Gaussian())      
    if type == 'Gamma': model = sm.GLM(Y, X, family=sm.families.Gamma(sm.families.links.Log()))
    return model

def CrossValidation(X, Y, type, split_perc = 0.75):
    split_size = int(len(Y) * split_perc)
    indices = np.arange(len(Y))
    
    MSE = []
    MSE_min = 1e10
    
    for i in range(1000):
        np.random.shuffle(indices)
        
        train_indices = indices[:split_size]
        test_indices = indices[split_size:]

        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        Y_train, Y_test = Y.iloc[train_indices], Y.iloc[test_indices]
        
        model = FitModel(X_train,Y_train,type)
        result = model.fit()
        Y_test_pred = result.predict(X_test)
        
        mse = np.mean((Y_test_pred - Y_test) ** 2)
        
        MSE.append(mse)
        
        if mse < MSE_min:  
            result_valid = result
            MSE_min = mse
    
    return result_valid, np.mean(MSE)
        
    
def Model(X, Y, type = 'Poisson'):
    
    result, average_mse = CrossValidation(X, Y, type)
    y_pred = result.predict(X)
    
    '''model = FitModel(X,Y,type)
    result = model.fit()
    y_pred = result.predict(X)
    average_mse = np.mean((y_pred - Y) ** 2)'''
    
    return result, y_pred, average_mse

def PlotModelPrediction(obs, pred, predictor, TYPE, title):
    fig, axs = plt.subplots(2, 1, figsize=(15, 5), sharex=True)
    axs[0].plot(obs.to_numpy(), color = 'black')
    axs[0].set_xticks([]) 
    axs[0].set_facecolor('white') 
    axs[0].set_ylabel(predictor, fontsize = 12)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)

    #axs[1].plot(pred.to_numpy(), color = 'blue', label = 'Correlation: '+str(round(np.corrcoef(obs, pred)[0,1],4)))
    axs[1].plot(pred, color = 'blue', label = 'Correlation: '+str(round(np.corrcoef(obs, pred)[0,1],4)))
    axs[1].set_xticks([]) 
    axs[1].set_facecolor('white') 
    axs[1].set_ylabel(predictor + ' Pred.', fontsize = 14)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].legend()
        
    plt.savefig('../Images/Social_Regression/'+ title + "/" + TYPE + '_Prediction.png')
    plt.show()


def PlotModelPrediction_Residual(obs, pred, predictor, TYPE, title):
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    x = obs.to_numpy()
    #y = pred.to_numpy()
    y = pred
    axs.scatter(x, y-x, s = 10)
    
    x_ = np.arange(0,2501,100)
    #axs.plot(x_, x_, color = 'red', linestyle = ':', linewidth = 1, label = 'y = x')
    
    axs.set_xlabel('Observed '+ predictor[0].upper() + predictor[1:] + ' (mm)', fontsize = 24)
    axs.set_ylabel('Residuals (mm)', fontsize = 24)
    #axs.set_aspect('equal', adjustable='box')
    #axs.set_xlim(0,2500)
    #axs.set_ylim(0,2500)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    #axs.legend(fontsize = 20)

    plt.tight_layout()
    plt.savefig('../Images/Social_Regression/' + title + "/" + TYPE + '_Prediction_Scatter.png')
    plt.show()

def PrintModelSummary(result, TYPE, title):
    fig, axs = plt.subplots(figsize=(20, 8))
    axs.axis('off')
    axs.text(0.5, 0.5, str(result.summary()),
                verticalalignment='center', horizontalalignment='left',
                transform=axs.transAxes, fontsize=12)
    plt.savefig('../Images/Social_Regression/' + title + "/" + TYPE + '_Model_Summary.png')
    plt.show()   



def Fit_Models(Visits, FEATURES, MODELS, PREDICTOR, type, mouse):
    X, Y = Variables(Visits, feature = FEATURES, predictor=PREDICTOR)
    
    for MODEL in MODELS:
        result, y_pred, average_mse = Model(X, Y, type = MODEL)
        print("Average MSE per Prediction for " + type + "-" + mouse + " " + MODEL + " Model Fitted: ", average_mse)

        PlotModelPrediction(Y, y_pred, predictor=PREDICTOR, TYPE = MODEL, title = type + "-" + mouse)
        PlotModelPrediction_Residual(Y, y_pred, predictor=PREDICTOR, TYPE = MODEL, title = type + "-" + mouse)
        PrintModelSummary(result, MODEL, title = type + "-" + mouse)

def main():
    #Process_Visits()
    #Visits = ConcatenateSessions()
    #Fit_Models(Visits, FEATURES = ['speed','acceleration', 'last_pellets_self', 'last_pellets_other', 'interval' , 'last_pellets_interval', 'entry'], MODELS = ['Poisson', 'Gaussian', 'Gamma'], PREDICTOR = 'duration')
    for i in range(len(LABELS)):
        type, mouse = LABELS[i][0], LABELS[i][1]
        Visits = pd.read_parquet('../SocialData/VisitData/'  + type + "_" + mouse +'_Visit.parquet', engine='pyarrow')
        
        Visits = Visits.dropna(subset=['speed'])
        Visits['distance'] = abs(Visits['distance'])
        Visits = Visits[Visits['distance'] >= 0.1]
        
        Fit_Models(Visits, FEATURES = ['last_pellets_self', 'last_pellets_other', 'interval' , 'last_duration', 'last_pellets_interval', 'entry'], MODELS = ['Poisson','Gaussian'], PREDICTOR = 'duration', type = type, mouse = mouse)    
if __name__ == "__main__":
    main()