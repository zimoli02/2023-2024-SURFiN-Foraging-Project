import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from pathlib import Path

import sys
from pathlib import Path

aeon_mecha_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(aeon_mecha_dir))

import Functions.kinematics as kinematics

from sklearn.cluster import KMeans


nodes_name = ['nose', 'head', 'right_ear', 'left_ear', 'spine1', 'spine2','spine3', 'spine4']

INFO = pd.read_parquet('../SocialData/INFO3.parquet', engine='pyarrow')
TYPE = ['Pre','Post']
MOUSE = ['BAA-1104045', 'BAA-1104047']



def DrawBody(data_x, data_y, axs):
    for k in range(len(nodes_name)): 
        axs.scatter(data_x[nodes_name[k]], data_y[nodes_name[k]])
    axs.plot([data_x['nose'],data_x['head']], [data_y['nose'],data_y['head']])
    axs.plot([data_x['left_ear'],data_x['nose']], [data_y['left_ear'],data_y['nose']])
    axs.plot([data_x['nose'],data_x['right_ear']], [data_y['nose'],data_y['right_ear']])
    axs.plot([data_x['left_ear'],data_x['right_ear']], [data_y['left_ear'],data_y['right_ear']])
    axs.plot([data_x['head'],data_x['spine1']], [data_y['head'],data_y['spine1']])
    axs.plot([data_x['spine1'],data_x['spine2']], [data_y['spine1'],data_y['spine2']])
    axs.plot([data_x['spine2'],data_x['spine3']], [data_y['spine2'],data_y['spine3']])
    axs.plot([data_x['spine3'],data_x['spine4']], [data_y['spine3'],data_y['spine4']])
    x_min, y_min = min(np.array(data_x)), min(np.array(data_y))
    axs.set_aspect('equal', 'box')
    axs.set_xlim(x_min-20, x_min+50)
    axs.set_ylim(y_min-20, y_min+50)
    return axs

def DrawPoses(center, d, data_x, data_y, axs):
    for j in range(len(center)):
        for i in range(len(data_x)):
            if abs(d[i] - center[j]) < 0.1: 
                if np.any(np.isnan(np.array(data_x.iloc[i]))): continue
                axs[j] = DrawBody(data_x.iloc[i],data_y.iloc[i], axs[j])
                axs[j].set_title(str(round(center[j],2)))
                break
    return axs

def BodyLength(type, mouse, data_x, data_y):
    dx = data_x['spine4'] - data_x['head']
    dy = data_y['spine4'] - data_y['head']
    d = np.sqrt(dx**2 + dy**2)

    data = np.array(d.dropna()).reshape(-1, 1)
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(data)
    
    center = np.sort(kmeans.cluster_centers_.T[0])
    fig, axs = plt.subplots(1,len(center), figsize = (len(center)*5,4))
    axs = DrawPoses(center, d, data_x, data_y, axs)
    plt.savefig('../Images/Social_BodyLength/' + type + "_" + mouse + '.png')
    plt.show()


def BodyAngle(type, mouse, data_x, data_y):
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
    
    data = degree_theta[~np.isnan(degree_theta)].reshape(-1,1)
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(data)

    center = np.sort(kmeans.cluster_centers_.T[0])
    fig, axs = plt.subplots(1,len(center), figsize = (len(center)*5,4))
    axs = DrawPoses(center, degree_theta, data_x, data_y, axs)
    plt.savefig('../Images/Social_BodyAngle/' + type + "_" + mouse + '.png')
    plt.show()


def Sniffing(type, mouse, data_x, data_y):
    mid_x = (data_x['right_ear'] + data_x['left_ear'])/2
    mid_y = (data_y['right_ear'] + data_y['left_ear'])/2
    dx = data_x['nose'] - mid_x
    dy = data_y['nose'] - mid_y
    d = np.sqrt(dx**2 + dy**2)

    data = np.array(d.dropna()).reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(data)
    
    center = np.sort(kmeans.cluster_centers_.T[0])
    fig, axs = plt.subplots(1,len(center), figsize = (len(center)*5,4))
    axs = DrawPoses(center, d, data_x, data_y, axs)
    plt.savefig('../Images/Social_Nose/' + type + "_" + mouse + '.png')
    plt.show()



def Get_Data(Type, Mouse):
    data_x, data_y= [], []
    for i in range(len(INFO)):
        type, mouse, start = INFO.loc[i, 'Type'], INFO.loc[i, 'Mouse'], INFO.loc[i, 'Start']
        x = pd.read_parquet('../SocialData/RawData/' + start + '_x.parquet', engine='pyarrow')
        y = pd.read_parquet('../SocialData/RawData/' + start + '_y.parquet', engine='pyarrow')
        if mouse == Mouse and type == Type: 
            data_x.append(x)
            data_y.append(y)
    
    data_x = pd.concat(data_x, ignore_index=False)
    data_y = pd.concat(data_y, ignore_index=False)
    
    data_x.to_parquet('../SocialData/BodyData/' + Type + "_" + Mouse + '_x.parquet', engine='pyarrow')
    data_y.to_parquet('../SocialData/BodyData/' + Type + "_" + Mouse + '_y.parquet', engine='pyarrow')

    return data_x, data_y

def Process_Pose(get_body_length, get_body_angle, get_nose_head_distance):
    for type in TYPE:
        for mouse in MOUSE:
            try:
                data_x = pd.read_parquet('../SocialData/BodyData/' + type + "_" + mouse + '_x.parquet', engine='pyarrow')
                data_y = pd.read_parquet('../SocialData/BodyData/' + type + "_" + mouse + '_y.parquet', engine='pyarrow')
            except FileNotFoundError:
                data_x, data_y = Get_Data(type, mouse)
            
            if get_body_length: BodyLength(type, mouse, data_x, data_y)
            if get_body_angle: BodyAngle(type, mouse, data_x, data_y)
            if get_nose_head_distance: Sniffing(type, mouse, data_x, data_y)
                

def main():
    

    get_body_length = False
    get_body_angle = False
    get_nose_head_distance = True

    Process_Pose(get_body_length,
    get_body_angle,
    get_nose_head_distance)

        

if __name__ == "__main__":
        main()