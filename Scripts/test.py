import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

import sys
from pathlib import Path
current_script_path = Path(__file__).resolve()

functions_dir = current_script_path.parents[1] / 'Functions'
sys.path.insert(0, str(functions_dir))
import mouse as mouse

def main():
    '''
    Mouse = mouse.Mouse(aeon_exp = 'AEON3', type = 'Pre', mouse = 'BAA-1104045')
    Mouse.Run_Visits()

    NODES = [['head', 'spine3'],['spine1', 'spine3'],['left_ear', 'spine3'],['right_ear', 'spine3']]
    for nodes in NODES:
        Mouse.Add_Body_Info_to_mouse_pos(property = 'distance', nodes = nodes)

    start = Mouse.active_chunk[0]
    mouse_pos = Mouse.mouse_pos[start:start + pd.Timedelta('3H')][::10]
    
    N = np.arange(3,50,1)
    Loglikelihood = np.zeros(len(N))
    
    # Run HMM fitting in parallel on multiple GPUs
    for i in range(len(N)):
        n= N[i]
        Mouse.hmm.Fit_Model_without_Saving(n_state=n, feature='Kinematics_and_Body')
        Loglikelihood[i] = Mouse.hmm.loglikelihood
        np.save('Likelihood.npy', Loglikelihood)
        print('End Inference for n = ', str(n))
'''
    N = np.arange(3,50,1)[:42]
    Loglikelihood = np.load('Likelihood.npy',allow_pickle=True)[:42]

    df = Loglikelihood[1:] - Loglikelihood[:-1]
    
    fig, axs = plt.subplots(1,2,figsize = (20,8))
    axs[0].scatter(N, Loglikelihood)
    axs[0].plot(N, Loglikelihood)
    axs[0].set_xticks(N)
    axs[1].scatter(N[1:], df)
    axs[1].plot(N[1:], df)
    axs[1].set_xticks(N[1:])
    for i in range(2):
        axs[i].axvline(x=10, color = 'red', linestyle = "--")
        axs[i].set_xlabel('State Number', fontsize = 20)
        axs[i].tick_params(axis='both', which='major', labelsize=12)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
    axs[0].set_ylabel('Log Likelihood per Point', fontsize = 20)
    axs[1].set_ylabel('$\Delta$Log Likelihood per Point', fontsize = 20)
    plt.tight_layout()
    plt.savefig('LogLikelihood.png')

    '''x, y, speed, acce, r, spine1_spine3, head_spine3, right_ear_spine3, left_ear_spine3 = [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)], [np.array([]) for _ in range(N)]
    for i in range(N):
        x[i] =  mouse_pos['smoothed_position_x'][states==i]
        y[i] = mouse_pos['smoothed_position_y'][states==i]
        speed[i] = mouse_pos['smoothed_speed'][states==i]
        acce[i] = mouse_pos['smoothed_acceleration'][states == i]
        r[i] = mouse_pos['r'][states == i]
        spine1_spine3[i] = mouse_pos['spine1-spine3'][states == i]
        head_spine3[i] = mouse_pos['head-spine3'][states == i]
        right_ear_spine3[i] = mouse_pos['right_ear-spine3'][states == i]
        left_ear_spine3[i] = mouse_pos['left_ear-spine3'][states == i]


    DATA = [speed, acce, head_spine3] 
    FEATURE = ['Speed', 'Acceleration', 'Body Length']
    fig, axs = plt.subplots(len(DATA), 1, figsize = (10, 20))
    for data, i in zip(DATA, range(len(DATA))):
        means = [np.mean(arr) for arr in data]
        var = [np.std(arr)/np.sqrt(len(arr)) for arr in data]
        axs[i].bar(range(N), means, yerr=var, capsize=5)
        axs[i].set_xticks(range(0, N), [str(j+1) for j in range(N)])
        axs[i].set_ylabel(FEATURE[i], fontsize = 20)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].tick_params(axis='both', which='major', labelsize=14)
    axs[2].set_xlabel('State', fontsize = 20)
    plt.tight_layout()
    plt.savefig('feature.png')'''
    
    
    
if __name__ == "__main__":
    main()
        