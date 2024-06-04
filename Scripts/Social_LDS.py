import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import sys
from pathlib import Path

current_script_path = Path(__file__).resolve()
function_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(function_dir))
import Functions.kinematics as kinematics
import Functions.inference as inference

parent_dir = current_script_path.parents[2] / 'aeon_mecha' 
sys.path.insert(0, str(parent_dir))
import aeon
import aeon.io.api as api
from aeon.schema.dataset import exp02
from aeon.analysis.utils import visits
from aeon.schema.schemas import social02

INFO = pd.read_parquet('../SocialData/INFO3.parquet', engine='pyarrow')
TYPE = ['Pre','Post']
MOUSE = ['BAA-1104045', 'BAA-1104047']
LABELS = [
    ['Pre','BAA-1104045'],
    ['Pre','BAA-1104047'],
    ['Post','BAA-1104045'],
    ['Post','BAA-1104047']
]

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
            

def Display_Along_Time():
    for label in LABELS:
        type, mouse = label[0], label[1]
        mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')

        start = mouse_pos.index[0]
        end = mouse_pos.index[-1]
        starts, ends = [],[]
        while start < end:
            end_ = start + pd.Timedelta('4H')

            starts.append(start)
            ends.append(end_)
            start = end_ + pd.Timedelta('1S')

        n = len(starts)
        fis, axs = plt.subplots(n, 4, figsize = (30, 4*n))
        for i in range(n):
            df = mouse_pos[starts[i]:ends[i]]
            speed = df.smoothed_speed
            axs[i,0].hist(speed[speed<10], density = True, bins = 100)
            axs[i,0].set_xlim((-0.5,10))
            axs[i,0].set_ylim((0,5))
            axs[i,1].hist(speed[speed>10], density = True, bins = 100)
            axs[i,1].set_xlim((9.5,1000))
            axs[i,1].set_ylim((0,0.06))
            
            acce = df.smoothed_acceleration
            axs[i,2].hist(acce[acce<5], density = True, bins = 100)
            axs[i,2].set_xlim((-0.5,5))
            axs[i,2].set_ylim((0,2))
            axs[i,3].hist(acce[acce>5], density = True, bins = 100)
            axs[i,3].set_xlim((4.5,2000))
            axs[i,3].set_ylim((0, 0.075))
            
            axs[i,0].set_ylabel(starts[i].hour)
        
        plt.savefig('../Images/Social_LDS/' + type + '_' + mouse +'_Kinematics_with_Time.png')
        plt.show()

def main():
    
    
    sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, V0, Z, R = kinematics.LDSParameters_Manual(dt=0.1)
    np.savez('../SocialData/LDS_Parameters/ManualParameters.npz', sigma_a = sigma_a, sigma_x = sigma_x, sigma_y = sigma_y, sqrt_diag_V0_value = sqrt_diag_V0_value, B = B, Qe = Qe, m0 = m0, V0 = V0, Z = Z, R = R)
    

        
    #ProcessLDSData()
    #Get_Parameters()
    #Get_Inference()
    #Display()
    
    Display_Along_Time()
        

if __name__ == "__main__":
        main()