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

import Functions.kinematics as kinematics


root = [Path("/ceph/aeon/aeon/data/raw/AEON2/experiment0.2")]
subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])

short_sessions = sessions.iloc[[4,16,17,20,23,24,25,26,28,29,30,31]]
long_sessions = sessions.iloc[[8, 10, 11, 14]]


        
def main():
    start = pd.Timestamp('2022-04-20 11:51:24.997630119')
    end = pd.Timestamp('2022-04-28 10:02:56.655809879')
    mouse_pos = api.load(root, exp02.CameraTop.Position, start=start, end=end)
            
    mouse_pos = kinematics.ProcessRawData(mouse_pos, root, start, end, exclude_maintenance=True, fix_nan=False, fix_nest=False)

            
    df = mouse_pos.copy()
    nan_blocks = df['x'].isna()

    for group, data in mouse_pos[nan_blocks].groupby((nan_blocks != nan_blocks.shift()).cumsum()):
        print(data.index[0], data.index[-1])
        duration = (data.index[-1] - data.index[0]).total_seconds()
                
        latest_valid_index = mouse_pos.loc[:data.index[0]-pd.Timedelta('0.018S'), 'x'].last_valid_index()
        if latest_valid_index is None:
            df.to_parquet('../Data/RawMouseKinematics/' + 'LongSession2' + 'mousepos.parquet', engine='pyarrow')
            print("latest_valid_index is None")
            break
        latest_valid_values = mouse_pos.loc[latest_valid_index, ['x', 'y']].values
            
        if len(data) == 1:
            df.loc[data.index, 'x'] = latest_valid_values[0]
            df.loc[data.index, 'y'] = latest_valid_values[1]
                
        else:    
            next_valid_index = mouse_pos.loc[data.index[-1]+pd.Timedelta('0.018S'):].first_valid_index()
            if next_valid_index is None: 
                df.to_parquet('../Data/RawMouseKinematics/' + 'LongSession2' + 'mousepos.parquet', engine='pyarrow')
                print("next_valid_index is None")
                break
            next_valid_values = mouse_pos.loc[next_valid_index, ['x', 'y']].values
                    
            interpolated_times = (data.index - latest_valid_index).total_seconds() / duration
                            
            total_x = next_valid_values[0] - latest_valid_values[0]
            total_y = next_valid_values[1] - latest_valid_values[1]
                            
            df.loc[data.index, 'x'] = latest_valid_values[0] + interpolated_times * total_x
            df.loc[data.index, 'y'] = latest_valid_values[1] + interpolated_times * total_y
    
    df.to_parquet('../Data/RawMouseKinematics/' + 'LongSession2' + 'mousepos.parquet', engine='pyarrow')

if __name__ == "__main__":
        main()