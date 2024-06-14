import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

import scipy.stats as stats
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

STARTS = [
    [pd.Timestamp('2024-01-31 11:28:39.00'), pd.Timestamp('2024-02-01 22:36:47.00'), pd.Timestamp('2024-02-02 00:15:00.00')],
    [pd.Timestamp('2024-02-05 15:43:07.00')],
    [pd.Timestamp('2024-02-25 17:22:33.00')],
    [pd.Timestamp('2024-02-28 13:54:17.00'), pd.Timestamp('2024-03-01 16:46:17.520991801')]
]

def inverse_sigmoid(y, midpoint, scale):
    y = Normalization(y)
    x = - scale * np.log(1 / y - 1) + midpoint
    return x

def Normalization(x):
    if min(x) <= 0 or max(x) >= 1:
        min_val = np.min(x) - 1e-8
        max_val = np.max(x) + 1e-8
        x = (x - min_val) / (max_val - min_val)
    return x

def sigmoid(x, midpoint, scale):
    return 1 / (1 + np.exp(-(x - midpoint) / scale))

def ConcatenateSessions():
    dfs = []
    for i in range(len(LABELS)):
        type, mouse = LABELS[i][0], LABELS[i][1]
        Visits = pd.read_parquet('../SocialData/VisitData/'  + type + "_" + mouse +'_Visit.parquet', engine='pyarrow')
        #Visits = Visits[Visits['duration'] <= 300]
        dfs.append(Visits)
        
    VISIT = pd.concat(dfs, ignore_index=True)
    VISIT = VISIT.sort_values(by='start',ignore_index=True) 
    
    return VISIT

def Process_Visits():
    pre_period_seconds = 10
    arena_r = 511
    for i in range(len(LABELS)):
        type, mouse = LABELS[i][0], LABELS[i][1]
        mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
        
        VISITS = []
        
        starts = STARTS[i]
        for j in range(len(starts)):
            if j == len(starts) - 1: mouse_pos_ = mouse_pos[starts[j]:]
            else: mouse_pos_ = mouse_pos[starts[j]:starts[j+1]]

            try:
                Visits_Patch1 = pd.read_parquet('../SocialData/VisitData/'  + type + "_" + mouse + "_" + str(j) + '_Visit1.parquet', engine='pyarrow')
            except FileNotFoundError:
                Visits_Patch1 = patch.Social_Visits(mouse_pos_, patch = 'Patch1', pre_period_seconds = pre_period_seconds, arena_r = arena_r)
                Visits_Patch1.to_parquet('../SocialData/VisitData/'  + type + "_" + mouse + "_" + str(j) + '_Visit1.parquet', engine='pyarrow')
            
            try:
                Visits_Patch2 = pd.read_parquet('../SocialData/VisitData/'  + type + "_" + mouse + "_" + str(j) + '_Visit2.parquet', engine='pyarrow')
            except FileNotFoundError:
                Visits_Patch2 = patch.Social_Visits(mouse_pos_, patch = 'Patch2', pre_period_seconds = pre_period_seconds, arena_r = arena_r)
                Visits_Patch2.to_parquet('../SocialData/VisitData/'  + type + "_" + mouse + "_" + str(j) + '_Visit2.parquet', engine='pyarrow')
            
            try:
                Visits_Patch3 = pd.read_parquet('../SocialData/VisitData/'  + type + "_" + mouse + "_" + str(j) + '_Visit3.parquet', engine='pyarrow')
            except FileNotFoundError:
                Visits_Patch3 = patch.Social_Visits(mouse_pos_, patch = 'Patch3', pre_period_seconds = pre_period_seconds, arena_r = arena_r)
                Visits_Patch3.to_parquet('../SocialData/VisitData/'  + type + "_" + mouse + "_" + str(j) + '_Visit3.parquet', engine='pyarrow')
        
            Visits = patch.VisitIntervals([Visits_Patch1, Visits_Patch2, Visits_Patch3]) 
            VISITS.append(Visits)
            
        VISITS = pd.concat(VISITS, ignore_index=True)
        VISITS = VISITS.sort_values(by='start',ignore_index=True) 
        VISITS.to_parquet('../SocialData/VisitData/'  + type + "_" + mouse +'_Visit.parquet', engine='pyarrow')
        print('Visits for ' + type + '-' + mouse + " Completed")

def Variables(VISIT, feature, predictor = 'distance'):
    X = VISIT[feature]
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), index = X.index, columns = X.columns)
    X['interc'] = 1
    Y = VISIT[[predictor]]
    
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

def CrossValidation(X, Y, type, midpoint, scale, split_perc = 0.75):    
    split_size = int(len(Y) * split_perc)
    indices = np.arange(len(Y))
    
    corre_max = -1
    for i in range(100):
        np.random.shuffle(indices)
        
        train_indices = indices[:split_size]
        test_indices = indices[split_size:]

        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        Y_train, Y_test = Y.iloc[train_indices], Y.iloc[test_indices]
        
        model = FitModel(X_train,Y_train,type)
        result = model.fit()
        Y_test_pred = result.predict(X_test)
        
        Y_test_pred = inverse_sigmoid(Y_test_pred.to_numpy(), midpoint=midpoint, scale=scale)
        Y_test = inverse_sigmoid(Y_test.to_numpy().reshape(1,-1)[0], midpoint=midpoint, scale=scale)
        
        #mse = np.mean((Y_test_pred.to_numpy() - Y_test.to_numpy()) ** 2)
        corre = np.corrcoef(Y_test_pred, Y_test)[0,1]
        if corre > corre_max:  
            result_valid = result
            corre_max = corre
    
    return result_valid, corre_max
        
    
def Model(X, Y, midpoint, scale, type = 'Poisson'):
    
    result, corre_max = CrossValidation(X, Y, type, midpoint, scale)
    y_pred = result.predict(X)

    return result, y_pred, corre_max

def PlotModelPrediction(obs, pred, predictor, TYPE, title):
    N = np.arange(0, len(pred), 1)
    '''obs = obs.to_numpy().reshape(1,-1)[0]
    pred = pred.to_numpy().reshape(1,-1)[0]'''
    
    fig, axs = plt.subplots(1, 1, figsize=(40, 3), sharex=True)
    axs.plot(N, obs, color = 'black')
    axs.set_xticks([]) 
    axs.set_facecolor('white') 
    axs.set_ylabel(predictor, fontsize = 12)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    axs_ = axs.twinx()
    axs_.plot(N, pred, color = 'blue', label = 'Correlation: '+str(round(np.corrcoef(obs, pred)[0,1],4)))
    axs_.set_xticks([]) 
    axs_.set_facecolor('white') 
    axs_.set_ylabel(predictor + ' Pred.', fontsize = 14)
    axs_.spines['top'].set_visible(False)
    axs_.spines['left'].set_visible(False)
    axs_.legend()
    
    #axs_.set_ylim(axs.get_ylim())
        
    plt.savefig('../Images/Social_Regression/'+ title + "/" + TYPE + '_Prediction.png')
    plt.show()



def PlotModelPrediction_Residual(obs, pred, predictor, TYPE, title):
    '''obs = obs.to_numpy().reshape(1,-1)[0]
    pred = pred.to_numpy().reshape(1,-1)[0]'''
    
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    axs.scatter(obs, pred-obs, s = 10)    
    axs.set_xlabel('Observed '+ predictor[0].upper() + predictor[1:] + ' (mm)', fontsize = 24)
    axs.set_ylabel('Residuals (mm)', fontsize = 24)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    #axs.legend(fontsize = 20)
    axs.set_ylim((-max(obs)-1, max(obs)+1))

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
        '''corre_max = -1
        for i in range(195, 205):
            for j in range(5, 50, 5):
                Y_ = sigmoid(Y, midpoint = i, scale = j)
                result, y_pred, corre = Model(X, Y_, midpoint = i, scale = j, type = MODEL)

                if corre > corre_max:  
                    result_valid = result
                    y_pred_valid = y_pred
                    midpoint_valid = i 
                    scale_valid = j
                    corre_max = corre
            print(midpoint_valid, scale_valid, corre_max)
        ''' 
        #y_pred =  np.exp(y_pred) - 1 
        midpoint_valid = 5 
        scale_valid = 100
        Y_ = sigmoid(Y, midpoint = midpoint_valid, scale = scale_valid)
        result_valid, y_pred_valid, corre = Model(X, Y_, midpoint = midpoint_valid, scale = scale_valid, type = MODEL)
        
        Y_ = inverse_sigmoid(Y_.to_numpy().reshape(1,-1)[0], midpoint=midpoint_valid, scale=scale_valid)
        y_pred_valid = inverse_sigmoid(y_pred_valid.to_numpy().reshape(1,-1)[0], midpoint=midpoint_valid, scale=scale_valid)
        
        PlotModelPrediction(Y_, y_pred_valid, predictor=PREDICTOR, TYPE = MODEL, title = type + "-" + mouse)
        PlotModelPrediction_Residual(Y_, y_pred_valid, predictor=PREDICTOR, TYPE = MODEL, title = type + "-" + mouse)
        PrintModelSummary(result_valid, MODEL, title = type + "-" + mouse)
        print('Midpoint:', midpoint_valid)
        print('Scale:', scale_valid)
        print('Correlation:', corre)
        np.save('Y.npy', Y_)
        np.save('Y_pred.npy', y_pred_valid)

def main():
    '''Process_Visits()
    print('Process Visits Completed')
    
    Visits = ConcatenateSessions()
    print('Concatenate Sessions Completed')
    
    Fit_Models(Visits, FEATURES = ['speed','acceleration','bodylength','bodyangle','nose','last_pellets_self', 'last_pellets_other','last_duration', 'last_interval','last_pellets_interval', 'entry'], MODELS = ['Gaussian'], PREDICTOR = 'duration', type = 'All', mouse = 'Mice')
    print('Fit Models for Concatenate Sessions Completed')'''
    
    for i in range(len(LABELS)):
        #if i != 0: break
        type, mouse = LABELS[i][0], LABELS[i][1]
        Visits = pd.read_parquet('../SocialData/VisitData/'  + type + "_" + mouse +'_Visit.parquet', engine='pyarrow')
        
        #Visits = Visits[Visits['duration'] <= 300]
        Visits = Visits.sort_values(by='start',ignore_index=True)  
        
        Fit_Models(Visits, FEATURES = ['speed','acceleration','bodylength','bodyangle','nose','last_pellets_self', 'last_pellets_other','last_duration', 'last_interval','last_pellets_interval', 'entry'], MODELS = ['Poisson', 'Gaussian'], PREDICTOR = 'duration', type = type, mouse = mouse)
        print('Fit Models for ' + type + '-' + mouse + " Completed")
if __name__ == "__main__":
    main()