import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pathlib import Path

import sys
from pathlib import Path

aeon_mecha_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(aeon_mecha_dir))

import aeon
import aeon.io.api as api
from aeon.schema.dataset import exp02
from aeon.analysis.utils import visits

import Functions.HMM as HMM
import Functions.kinematics as kinematics
import Functions.patch as patch


nodes_name = ['nose', 'head', 'right_ear', 'left_ear', 'spine1', 'spine2','spine3', 'spine4']

INFO = pd.read_parquet('../SocialData/INFO3.parquet', engine='pyarrow')
TYPE = ['Pre','Post']
MOUSE = ['BAA-1104045', 'BAA-1104047']


#def Get_Latent_States():

def Get_Observations(Kinematics_Update = False,
                        Weight_Update = False,
                        r_Update = True):
    for type in TYPE:
        for mouse in MOUSE:
            try:
                mouse_pos = pd.read_parquet('../SocialData/HMMData/' + type + "_" + mouse + '.parquet', engine='pyarrow')
            except FileNotFoundError:
                mouse_pos = Get_Data(type, mouse)
            
            

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
    
    Get_Observations(Kinematics_Update = False,
                        Weight_Update = False,
                        r_Update = True)
        



if __name__ == "__main__":
        main()