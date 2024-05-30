import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pathlib import Path

import sys
from pathlib import Path

aeon_mecha_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(aeon_mecha_dir))

import Functions.HMM as HMM
import Functions.kinematics as kinematics
import Functions.patch as patch


nodes_name = ['nose', 'head', 'right_ear', 'left_ear', 'spine1', 'spine2','spine3', 'spine4']

INFO = pd.read_parquet('../SocialData/INFO3.parquet', engine='pyarrow')
TYPE = ['Pre']
MOUSE = ['BAA-1104045']
LABELS = [
    ['Pre','BAA-1104045']
]

color_names = ['black', "blue", "red", "tan", "green", "brown", "purple", "orange", 'turquoise', "black"]

def FixNan(mouse_pos, dt, column):
    df = mouse_pos.copy()
    nan_blocks = df[column].isna()

    for group, data in mouse_pos[nan_blocks].groupby((nan_blocks != nan_blocks.shift()).cumsum()):
        latest_valid_index = mouse_pos.loc[:data.index[0]-(pd.Timedelta(dt)-pd.Timedelta('0.002S')), column].last_valid_index()
        
        
        if latest_valid_index is None:
            first_valid_index = df[column].first_valid_index()
            first_valid_value = df.loc[first_valid_index, column]
            df.loc[data.index, column] = first_valid_value
        else:
            latest_valid_value = mouse_pos.loc[latest_valid_index, column]
        
            if len(data) == 1:
                df.loc[data.index, column] = latest_valid_value
                
            else:    
                next_valid_index = mouse_pos.loc[data.index[-1]+(pd.Timedelta(dt)-pd.Timedelta('0.002S')):].first_valid_index()
                next_valid_value = mouse_pos.loc[next_valid_index, column]
                
                duration = (data.index[-1] - latest_valid_index).total_seconds()
                interpolated_times = (data.index - latest_valid_index).total_seconds() / duration        
                total = next_valid_value - latest_valid_value    
                df.loc[data.index, column] = latest_valid_value + interpolated_times * total
    
    return df

'''
def Get_Latent_States(id, n, features):
    type, mouse = LABELS[id][0], LABELS[id][1]
    mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
    obs = np.array(mouse_pos[features])
    hmm, states, transition_mat, lls = HMM.FitHMM(obs, num_states = n, n_iters = 50)
    
    for label in LABELS:
        type, mouse = label[0], label[1]
        mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
        obs = np.array(mouse_pos[features])
        
        states = hmm.most_likely_states(obs)
        
        new_values = np.empty_like(states)
        for i, val in enumerate(index): new_values[states == val] = i
        states = new_values
        np.save('../SocialData/HMMStates/' + type + "_" + mouse + "_States.npy", states)


def Display_Latent_States(N):
    X, Y, SPEED, ACCE, LENGTH, ANGLE = [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)]
    for label in LABELS:
        type, mouse = label[0], label[1]
        mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
        states = np.load('../SocialData/HMMStates/' + type + "_" + mouse + "_States.npy", allow_pickle = True)
        mouse_pos['state'] = pd.Series(states, index = mouse_pos.index)

        for i in range(N):
            X[i] = np.concatenate([X[i], mouse_pos['smoothed_position_x'][states==i]])
            Y[i] = np.concatenate([Y[i], mouse_pos['smoothed_position_y'][states==i]])
            SPEED[i] = np.concatenate([SPEED[i], mouse_pos['smoothed_speed'][states==i]])
            ACCE[i] = np.concatenate([ACCE[i], mouse_pos['smoothed_acceleration'][states == i]])
            LENGTH[i] = np.concatenate([LENGTH[i], mouse_pos['bodylength'][states == i]])
            ANGLE[i] = np.concatenate([ANGLE[i], mouse_pos['bodyangle'][states == i]])
        
    fig, axs = plt.subplots(1, N, figsize = (N*8-2,6))
    for i in range(N):
        axs[i].scatter(X[i], Y[i], color = color_names[i], s = 2, alpha = 0.2)
        axs[i].set_xlim(145, 1250)
        axs[i].set_ylim(50, 1080)
        axs[i].set_title('State' + str(i))
        axs[i].set_xlabel('X')
        axs[i].set_ylabel('Y')
    plt.savefig('../Images/Social_HMM/Position.png')
    plt.show()
    
    DATA = [SPEED, ACCE, LENGTH, ANGLE]
    FEATURE = [SPEED, ACCE, LENGTH, ANGLE]
    fig, axs = plt.subplots(len(FEATURE), 1, figsize = (10, len(FEATURE)*7-1))
    for data, i in zip(DATA, range(len(DATA))):
        means = [np.mean(arr) for arr in data]
        var = [np.std(arr)/np.sqrt(len(arr)) for arr in data]
        axs[i].bar(range(N), means, yerr=var, capsize=5)
        axs[i].set_xticks(range(0, N), [str(j) for j in range(N)])
        axs[i].set_ylabel(FEATURE[i])
    plt.savefig('../Images/Social_HMM/Data.png')
    plt.show()
'''        

def Get_Latent_State_Number(features, N):
    type = 'Pre'
    mouse = 'BAA-1104045'

    mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')

    obs = np.array(mouse_pos[features])

    loglikelihood = []
    for n in N:
        hmm, states, transition_mat, lls = HMM.FitHMM(obs, num_states = n, n_iters = 50)
        loglikelihood.append(lls)
    
    np.save('../SocialData/HMMData/LogLikelihood_single.npy', loglikelihood)

def Display_Latent_State_Number(N):    
    LogLikelihood = np.load('../SocialData/HMMData/LogLikelihood_single.npy', allow_pickle=True)
    fig, axs = plt.subplots(1,1,figsize = (10,7))
    for i in range(len(LogLikelihood)):
        loglikelihood = LogLikelihood[i]
        axs.scatter(N, loglikelihood)
        axs.plot(N, loglikelihood, color = 'black', label = i)
    axs.set_xticks(N)
    axs.legend()
    plt.savefig('../Images/Social_HMM/StateNumber_Single.png')
    plt.show()

def Get_Observations(weight_Update, patch_Update, r_Update, bodylength_Update, bodyangle_Update, nose_Update):
    for type in TYPE:
        for mouse in MOUSE:
            try:
                mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
            except FileNotFoundError:
                mouse_pos = Get_Data(type, mouse)
            
            data_x = pd.read_parquet('../SocialData/BodyData/' + type + "_" + mouse + '_x.parquet', engine='pyarrow')
            data_y = pd.read_parquet('../SocialData/BodyData/' + type + "_" + mouse + '_y.parquet', engine='pyarrow')

            if patch_Update:
                mouse_pos = patch.InPatch(mouse_pos, r = 30, interval = 5, patch_loc = [])
            if r_Update:
                mouse_pos = patch.Radius(mouse_pos, x_o = 709.4869937896729, y_o = 546.518087387085)
            if bodylength_Update:
                dx = data_x['spine4'] - data_x['head']
                dy = data_y['spine4'] - data_y['head']
                d = np.sqrt(dx**2 + dy**2)
                mouse_pos['bodylength'] = pd.Series(d, index=mouse_pos.index)
                mouse_pos = FixNan(mouse_pos,'0.1S', 'bodylength')
            if bodyangle_Update:
                head = np.array([data_x['head'], data_y['head']]).T
                spine2 = np.array([data_x['spine2'], data_y['spine2']]).T
                spine4 = np.array([data_x['spine4'], data_y['spine4']]).T
                v1 = head - spine2
                v2 = spine4 - spine2
                dot_product = np.einsum('ij,ij->i', v1, v2)
                norm1 = np.linalg.norm(v1, axis=1)
                norm2 = np.linalg.norm(v2, axis=1)
                cos_theta = dot_product / (norm1 * norm2)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                radians_theta = np.arccos(cos_theta)
                degree_theta = np.degrees(radians_theta)
                mouse_pos['bodyangle'] = pd.Series(degree_theta, index=mouse_pos.index)
                mouse_pos = FixNan(mouse_pos,'0.1S', 'bodyangle')
            if nose_Update:
                dx = data_x['nose'] - data_x['head']
                dy = data_y['nose'] - data_y['head']
                d = np.sqrt(dx**2 + dy**2)
                mouse_pos['nose'] = pd.Series(d, index=mouse_pos.index)
                mouse_pos = FixNan(mouse_pos,'0.1S', 'nose')
                
            mouse_pos.to_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')

def Get_Data(Type, Mouse):
    Mouse_pos = []
    for i in range(len(INFO)):
        type, mouse, start = INFO.loc[i, 'Type'], INFO.loc[i, 'Mouse'], INFO.loc[i, 'Start']
        if mouse == Mouse and type == Type: 
            mouse_pos = pd.read_parquet('../SocialData/LDSData/' + start + '.parquet', engine='pyarrow')
            smoothRes = np.load('../SocialData/LDS/' + start +'_smoothRes.npz')
            kinematics.AddKinematics(smoothRes, mouse_pos)
            Mouse_pos.append(mouse_pos)
    
    Mouse_pos = pd.concat(Mouse_pos, ignore_index=False)
    Mouse_pos.to_parquet('../SocialData/HMMData/' + Type + "_" + Mouse + '.parquet', engine='pyarrow')

    return Mouse_pos

def main():
    
    features = ['smoothed_speed', 'smoothed_acceleration', 'r','bodylength', 'bodyangle', 'nose']
    
    '''
    Get_Observations(weight_Update = False,
                        patch_Update = False,
                        r_Update = False,
                        bodylength_Update = True,
                        bodyangle_Update = True,
                        nose_Update = True)
    '''
        
    Get_Latent_State_Number(features, N = np.arange(3,20))
    Display_Latent_State_Number(N = np.arange(3,20))
    
    """
    Get_Latent_States(id, n, features)
    Display_Latent_States()
    """
    


if __name__ == "__main__":
        main()