import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

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
from aeon.analysis.utils import visits, distancetravelled



def main():

    root = '/ceph/aeon/aeon/data/raw/AEON3/social0.2'
    mouse_pos = pd.read_parquet('../SocialData/HMMData/' + 'Pre' + "_" + 'BAA-1104045' + '.parquet', engine='pyarrow')

    start = mouse_pos.index[0]
    end = mouse_pos.index[-1]

    check_timestamp = pd.Timestamp('2024-02-03 07:59:59.980000')
    if start < check_timestamp < end:
        encoders = []
        starts = [start, pd.Timestamp('2024-02-03 08:00:01.000000')]
        ends = [pd.Timestamp('2024-02-03 07:59:58.000000'), end]
        
        for i in range(len(starts)):
            encoder = aeon.load(root, social02.Patch2.Encoder, start=starts[i], end=ends[i])
            encoders.append(encoder)
        encoder = pd.concat(encoders, ignore_index=False)
    else:
        encoder = aeon.load(root, social02.Patch2.Encoder, start=start, end=end)

    encoder = encoder[~encoder.index.duplicated(keep='first')]
    encoder = encoder[::5]
    encoder.to_parquet('encoder.parquet', engine='pyarrow')
    
    '''w = -distancetravelled(encoder.angle).to_numpy()
    dw = np.concatenate((np.array([0]), w[:-1]- w[1:]))
    encoder['Distance'] = pd.Series(w, index=encoder.index)
    encoder['DistanceChange'] = pd.Series(dw, index=encoder.index)
    encoder['DistanceChange'] = encoder['DistanceChange'].rolling('10S').mean()
    encoder['Move'] = np.where(abs(encoder.DistanceChange) > 0.001, 1, 0)
    
    encoder = encoder[~encoder.index.duplicated(keep='first')]
    
    encoder.to_parquet('encoder.parquet', engine='pyarrow')
    
    if interval_seconds < 0.01: return encoder
    groups = encoder['Move'].ne(encoder['Move'].shift()).cumsum()
    one_groups = encoder[encoder['Move'] == 1].groupby(groups).groups
    one_groups = list(one_groups.values())

    for i in range(len(one_groups) - 1):
        end_current_group = one_groups[i][-1]
        start_next_group = one_groups[i + 1][0]
        duration = start_next_group - end_current_group

        if duration < pd.Timedelta(seconds=interval_seconds):
            encoder.loc[end_current_group:start_next_group, 'Move'] = 1'''




if __name__ == "__main__":
        main()