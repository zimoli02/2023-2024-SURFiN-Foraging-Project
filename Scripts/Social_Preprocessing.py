import numpy as np
import pandas as pd

import sys
from pathlib import Path

import h5py
from datetime import datetime, timedelta

aeon_mecha_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(aeon_mecha_dir))


def DeleteNan(df, df_):
    temp_df = df.dropna(subset=['spine2'])
    first_valid_index, last_valid_index = temp_df.index[0], temp_df.index[-1]
    df = df.loc[first_valid_index:last_valid_index]
    df_ = df_.loc[first_valid_index:last_valid_index]
    
    temp_df = df_.dropna(subset=['spine2'])
    first_valid_index, last_valid_index = temp_df.index[0], temp_df.index[-1]
    df = df.loc[first_valid_index:last_valid_index]
    df_ = df_.loc[first_valid_index:last_valid_index]
    
    return df, df_

def main():
    INFO = pd.read_parquet('../SocialData/INFO3.parquet', engine='pyarrow')
    file_path = "/ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraTop/predictions_social02/AEON3/analyses/CameraTop_"
    nodes_name = ['nose', 'head', 'right_ear', 'left_ear', 'spine1', 'spine2','spine3', 'spine4']
    for i in range(len(INFO)):
        print(i)
        start, end = INFO.loc[i, 'Start'], INFO.loc[i, 'End']
        try:
            data = pd.read_parquet('../SocialData/RawData/' + start + '_x.parquet', engine='pyarrow')
        except FileNotFoundError:
            start_chunk, end_chunk = datetime.strptime(start, '%Y-%m-%dT%H-%M-%S'), datetime.strptime(end, '%Y-%m-%dT%H-%M-%S')
            chunk = start_chunk.replace(minute=0, second=0, microsecond=0)
            flag = 0
            chunks = []
            while flag == 0:
                chunk_time = chunk.strftime('%Y-%m-%dT%H-%M-%S')
                chunks.append(chunk_time)
                chunk += timedelta(hours=1)
                if chunk_time == end: flag = 1
            
            dfs_x, dfs_y = [], []
            for j in range(len(chunks)):
                file_path_ = file_path + chunks[j] + "_full_pose.analysis.h5"
                with h5py.File(file_path_, 'r') as f: tracks_matrix = f['tracks'][:]
                num_frames = tracks_matrix.shape[3]

                if j == 0: start_time = datetime.strptime(chunks[j+1], '%Y-%m-%dT%H-%M-%S') - timedelta(seconds=num_frames*0.02)
                else: start_time = datetime.strptime(chunks[j], '%Y-%m-%dT%H-%M-%S')

                timestamps = [start_time + timedelta(seconds=k*0.02) for k in range(num_frames)]
                
                df_x = pd.DataFrame(tracks_matrix[0][0].T, index=timestamps, columns=nodes_name)
                dfs_x.append(df_x)
                
                df_y = pd.DataFrame(tracks_matrix[0][1].T, index=timestamps, columns=nodes_name)
                dfs_y.append(df_y)
            
            data_x = pd.concat(dfs_x, ignore_index=False)
            data_y = pd.concat(dfs_y, ignore_index=False)
            data_x, data_y = data_x[::5],  data_y[::5]
            data_x, data_y = DeleteNan(data_x, data_y)
            
            data_x.to_parquet('../SocialData/RawData/' + start + '_x.parquet', engine='pyarrow')
            data_y.to_parquet('../SocialData/RawData/' + start + '_y.parquet', engine='pyarrow')
            
            


if __name__ == "__main__":
        main()