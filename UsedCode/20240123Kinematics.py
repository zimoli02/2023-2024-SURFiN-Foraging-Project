import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, SecondLocator, DateFormatter

import plotly.graph_objects as go
import plotly.express as px

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import scipy
import pyarrow

import inference

import aeon
import aeon.io.api as api
from aeon.io import reader, video
from aeon.schema.dataset import exp02
from aeon.analysis.utils import visits, distancetravelled

import kinematics
import patch


def main():
        root = [Path("/ceph/aeon/aeon/data/raw/AEON2/experiment0.2")]

        subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
        sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])
        long_sessions = sessions.iloc[[8, 10, 11, 14]]

        P = np.load('LongSession3Parameters.npz',allow_pickle=True)
        sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, V0, Z, R = P['sigma_a'].item(), P['sigma_x'].item(), P['sigma_y'].item(), P['sqrt_diag_V0_value'].item(), P['B'], P['Qe'], P['m0'], P['V0'], P['Z'], P['R']
        dt = 0.019968
        
        for session, i in zip(list(long_sessions.itertuples()), range(len(long_sessions))):
                if i == 3:
                        print(session.id)
                        title = 'LongSession'+str(i)
                        start, end = session.enter, session.exit
                        mouse_pos = api.load(root, exp02.CameraTop.Position, start=start, end=end)
                        
                        mouse_pos = kinematics.ProcessRawData(mouse_pos, root, start, end)

                        obs = np.transpose(mouse_pos[["x", "y"]].to_numpy())

                        Q = sigma_a**2*Qe

                        filterRes = inference.filterLDS_SS_withMissingValues_np(
                                y=obs, B=B, Q=Q, m0=m0, V0=V0, Z=Z, R=R)
                        np.savez_compressed(title + 'filterRes.npz', **filterRes)

                        smoothRes = inference.smoothLDS_SS( 
                                B=B, xnn=filterRes["xnn"], Vnn=filterRes["Vnn"],
                                xnn1=filterRes["xnn1"], Vnn1=filterRes["Vnn1"], m0=m0, V0=V0)
                        np.savez_compressed(title + 'smoothRes.npz', **smoothRes)
                        
                        '''
                        patch.AddKinematics(title, mouse_pos)

                        mouse_pos = mouse_pos[mouse_pos['smoothed_speed'] <= 2000]
                                
                        weight = api.load(root, exp02.Nest.WeightSubject, start=start, end=end)
                        patch.AddWeight(mouse_pos, weight)
                                
                        pellets_patch1 = api.load(root, exp02.Patch1.DeliverPellet, start=start, end=end)
                        pellets_patch2 = api.load(root, exp02.Patch2.DeliverPellet, start=start, end=end)
                        
                        patch.InPatch(mouse_pos, pellets_patch1, pellets_patch2)

                        mouse_pos.to_parquet(title+'mousepos.parquet', engine='pyarrow')

                        mouse_pos = mouse_pos.dropna(subset=['x'])
                        fig, axs = plt.subplots(4,1, figsize = (40,24))
                        mouse_pos.x.plot(ax = axs[0])
                        mouse_pos.y.plot(ax = axs[0])
                        mouse_pos.smoothed_position_x.plot(ax = axs[1])
                        mouse_pos.smoothed_position_y.plot(ax = axs[1])
                        mouse_pos.smoothed_speed.plot(ax = axs[2])
                        mouse_pos.smoothed_acceleration.plot(ax = axs[3])
                        plt.savefig('images/Kinematics/' + title+'.png')'''
                        
                        x  = [0,1,2,3,4]
                        fig, axs = plt.subplots(1,1)
                        axs.plot(x)
                        plt.savefig('images/Kinematics/'+title+'.png')

if __name__ == "__main__":
        main()