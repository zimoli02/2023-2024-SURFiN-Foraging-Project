import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import sys
from pathlib import Path

aeon_mecha_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(aeon_mecha_dir))

import aeon
import aeon.io.api as api
from aeon.schema.dataset import exp02
from aeon.analysis.utils import visits

import Functions.kinematics as kinematics
import Functions.inference as inference

INFO = pd.read_parquet('../SocialData/INFO3.parquet', engine='pyarrow')

def CenterOfMass(data_x, data_y):
    x = data_x['spine2']
    y = data_y['spine2']
    mouse_pos = pd.DataFrame({'x': x, 'y': y})

    return mouse_pos


def ProcessLDSData():
    for i in range(len(INFO)):
        start, end = INFO.loc[i, 'Start'], INFO.loc[i, 'End']
        try:
            mouse_pos = pd.read_parquet('../SocialData/LDSData/' + start + '.parquet', engine='pyarrow')
        except FileNotFoundError:
            data_x = pd.read_parquet('../SocialData/RawData/' + start + '_x.parquet', engine='pyarrow')
            data_y = pd.read_parquet('../SocialData/RawData/' + start + '_y.parquet', engine='pyarrow')
            
            mouse_pos = CenterOfMass(data_x, data_y)
            mouse_pos = kinematics.FixNan(mouse_pos, dt = '0.1S')
            
            mouse_pos.to_parquet('../SocialData/LDSData/' + start + '.parquet', engine='pyarrow')


def Infer_Parameters(start):
    try:
        mouse_pos = pd.read_parquet('../SocialData/LDSData/' + start + '.parquet', engine='pyarrow')
    except FileNotFoundError:
        ProcessLDSData()
    
    #First 20 min of the data, 10Hz
    obs = np.transpose(mouse_pos[["x", "y"]].to_numpy())[:, :20*60*10]
    np.save('../SocialData/LDS_Parameters/' + start + '_OBS.npy', obs)
    
    P = np.load('../SocialData/LDS_Parameters/ManualParameters.npz', allow_pickle=True)
    sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, V0, Z, R = P['sigma_a'].item(), P['sigma_x'].item(), P['sigma_y'].item(), P['sqrt_diag_V0_value'].item(), P['B'], P['Qe'], P['m0'], P['V0'], P['Z'], P['R']

    sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, m0, V0, Z, R = kinematics.LDSParameters_Learned(obs, sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, Z)
    np.savez('../SocialData/LDS_Parameters/' + start + '_Parameters.npz', sigma_a = sigma_a, sigma_x = sigma_x, sigma_y = sigma_y, sqrt_diag_V0_value = sqrt_diag_V0_value, B = B, Qe = Qe, m0 = m0, V0 = V0, Z = Z, R = R)
    

def Get_Parameters():
    for i in range(len(INFO)):
        start, end = INFO.loc[i, 'Start'], INFO.loc[i, 'End']
        try:
            P = np.load('../SocialData/LDS_Parameters/' + start + '_Parameters.npz', allow_pickle=True)
            obs = np.load('../SocialData/LDS_Parameters/' + start + '_OBS.npy', allow_pickle=True)
        except FileNotFoundError:
            Infer_Parameters(start)


def Inference(start):
    mouse_pos = pd.read_parquet('../SocialData/LDSData/' + start + '.parquet', engine='pyarrow')
    obs = np.transpose(mouse_pos[["x", "y"]].to_numpy())

    P = np.load('../SocialData/LDS_Parameters/' + start + '_Parameters.npz', allow_pickle=True)
    sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, V0, Z, R = P['sigma_a'].item(), P['sigma_x'].item(), P['sigma_y'].item(), P['sqrt_diag_V0_value'].item(), P['B'], P['Qe'], P['m0'], P['V0'], P['Z'], P['R']
    Q = (sigma_a**2) * Qe

    # Filtering
    filterRes = inference.filterLDS_SS_withMissingValues_np(
        y=obs, B=B, Q=Q, m0=m0, V0=V0, Z=Z, R=R)
    np.savez_compressed('../SocialData/LDS/' + start +'_filterRes.npz', **filterRes)
        
    # Smoothing
    smoothRes = inference.smoothLDS_SS( 
        B=B, xnn=filterRes["xnn"], Vnn=filterRes["Vnn"],
        xnn1=filterRes["xnn1"], Vnn1=filterRes["Vnn1"], m0=m0, V0=V0)
    np.savez_compressed('../SocialData/LDS/' + start +'_smoothRes.npz', **smoothRes) 


def Get_Inference():
    for i in range(len(INFO)):
        start, end = INFO.loc[i, 'Start'], INFO.loc[i, 'End']
        try:
            smoothRes = np.load('../SocialData/LDS/' + start +'_smoothRes.npz')
        except FileNotFoundError:
            Inference(start)


def Display():
    for i in range(len(INFO)):
        start, end = INFO.loc[i, 'Start'], INFO.loc[i, 'End']
        mouse_pos = pd.read_parquet('../SocialData/LDSData/' + start + '.parquet', engine='pyarrow')
        smoothRes = np.load('../SocialData/LDS/' + start +'_smoothRes.npz')
        kinematics.AddKinematics(smoothRes, mouse_pos)

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
        plt.savefig('../Images/Social_LDS/' + start +'.png')
        plt.show()
        
        '''fig, axs = plt.subplots(1,2,figsize = (20, 8))
        axs[0].plot(mouse_pos.x, mouse_pos.y, color = 'blue')
        axs[1].plot(mouse_pos.smoothed_position_x, mouse_pos.smoothed_position_y, color = 'blue')
        axs[0].set_xlabel('Raw Position', fontsize = 12)
        axs[1].set_xlabel('Smoothed Position', fontsize = 12)
        for i in range(2):
            axs[i].set_xlim(100, 1300)
            axs[i].set_ylim(0, 1150)
        plt.savefig('../Images/Positions/' + start +'.png')
        plt.show()'''
            

wo
def main():
    
    
    sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, V0, Z, R = kinematics.LDSParameters_Manual(dt=0.1)
    np.savez('../SocialData/LDS_Parameters/ManualParameters.npz', sigma_a = sigma_a, sigma_x = sigma_x, sigma_y = sigma_y, sqrt_diag_V0_value = sqrt_diag_V0_value, B = B, Qe = Qe, m0 = m0, V0 = V0, Z = Z, R = R)
    

        
    #ProcessLDSData()
    Get_Parameters()
    Get_Inference()
    Display()
        

if __name__ == "__main__":
        main()