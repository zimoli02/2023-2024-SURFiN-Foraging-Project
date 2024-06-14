import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter


import sys
from pathlib import Path
current_script_path = Path(__file__).resolve()

function_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(function_dir))
import Functions.mouse as mouse
from SSM.ssm.plots import gradient_cmap

parent_dir = current_script_path.parents[2] / 'aeon_mecha' 
sys.path.insert(0, str(parent_dir))
import aeon
import aeon.io.api as api
from aeon.io import reader, video
from aeon.schema.schemas import social02

LABELS = [
    ['Pre','BAA-1104045'],
    ['Pre','BAA-1104047'],
    ['Post','BAA-1104045'],
    ['Post','BAA-1104047']
]
nodes_name = ['nose', 'head', 'right_ear', 'left_ear', 'spine1', 'spine2','spine3', 'spine4']
color_names = ['black', "blue", "red", "tan", "green", "brown", "purple", "orange", 'turquoise', "yellow", 'pink', 'darkblue']


def main():
    N = np.arange(3,25)
    for label in LABELS:
        type_name, mouse_name = label[0], label[1]
        Mouse = mouse.Mouse(aeon_exp='AEON3', type = type_name, mouse = mouse_name)

        NODES = [['head', 'spine3'],['spine1', 'spine3'],['left_ear', 'spine3'],['right_ear', 'spine3']]
        for nodes in NODES:
            Mouse.Add_Body_Info_to_mouse_pos(property = 'distance', nodes = nodes)

        print('Start Inference')
        Loglikelihood = []
        for n in N:
            Mouse.hmm.model_period = Mouse.active_chunk
            Mouse.hmm.Fit_Model_without_Saving(n_state = n, feature = 'Kinematics_and_Body')
            
            Mouse.hmm.model_period = Mouse.active_chunk
            Loglikelihood.append(Mouse.hmm.loglikelihood)
            print('End Inference for n = ', str(n))
        
        fig, axs = plt.subplots(1,1,figsize = (10,7))
        axs.scatter(N, Loglikelihood)
        axs.plot(N, Loglikelihood, color = 'black')
        axs.set_xticks(N)
        plt.savefig('../Images/Social_HMM/StateNumber/' + type_name + '-' + mouse_name + '.png')


if __name__ == "__main__":
        main()