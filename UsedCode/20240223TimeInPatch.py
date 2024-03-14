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
from sklearn.preprocessing import StandardScaler

import kinematics
import patch

root = [Path("/ceph/aeon/aeon/data/raw/AEON2/experiment0.2")]


def main():
        subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
        sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])
        short_sessions = sessions.iloc[[4,16,17,20,23,24,25,26,28,29,30,31]]
        
        for session, i in zip(list(short_sessions.itertuples()), range(len(short_sessions))):
                title = 'ShortSession'+str(i)
                print(title)

                mouse_pos = pd.read_parquet(title+'mousepos.parquet', engine='pyarrow')
                mouse_pos = mouse_pos[mouse_pos['smoothed_acceleration'] <= 60000]
                mouse_pos = patch.DeleteRows(mouse_pos)
                
                states = np.load(title+"States.npy", allow_pickle = True)
                mouse_pos['states'] = pd.Series(states, index=mouse_pos.index)
                
                
                Visits_Patch1 = patch.Visits(mouse_pos, patch = 'Patch1', pre_period_seconds = 10)
                Visits_Patch2 = patch.Visits(mouse_pos, patch = 'Patch2', pre_period_seconds = 10)
                Visits_Patch1, Visits_Patch2 = patch.VisitIntervals(Visits_Patch1, Visits_Patch2) 

                Visits_Patch1.to_parquet(title+'Visit1.parquet', engine='pyarrow')
                Visits_Patch2.to_parquet(title+'Visit2.parquet', engine='pyarrow')
                
                VISIT = pd.concat([Visits_Patch1, Visits_Patch2], ignore_index=False)

                X = VISIT[['speed','acceleration', 'weight','PelletsInLastVisitSelf', 'PelletsInLastVisitOther', 'IntervalLastVisit' ,'entry']].to_numpy()
                Y = VISIT['distance'].to_numpy()

                scaler = StandardScaler()
                X = scaler.fit_transform(X)

        
                coefficients, intercept, y_pred, mse, r2 = patch.SimpleLinearRegression(X, Y)
                
                fig, axs = plt.subplots(2, 1, figsize=(15, 5), sharex=True)

                axs[0].plot(Y, color = 'black')
                axs[0].set_xticks([]) 
                axs[0].set_facecolor('white') 
                axs[0].set_ylabel('Dis.', fontsize = 12)
                axs[0].spines['top'].set_visible(False)
                axs[0].spines['right'].set_visible(False)

                axs[1].plot(y_pred, color = 'blue', label = 'r2='+str(round(r2, 3)))
                axs[1].set_xticks([]) 
                axs[1].set_facecolor('white') 
                axs[1].set_ylabel('Dis. Pred.', fontsize = 14)
                axs[1].spines['top'].set_visible(False)
                axs[1].spines['right'].set_visible(False)
                axs[1].legend()

                plt.savefig('/nfs/nhome/live/zimol/ProjectAeon/aeon_mecha/images/Regression/'+title+'Prediction.png')

                COFF = []
                for t in range (1000):
                        index = np.random.choice(np.arange(len(Y)), size=len(Y), replace=True)
                        coefficients1, intercept1, y1_pred, mse1, r21 = patch.SimpleLinearRegression(X[index], Y[index])
                        COFF.append(coefficients1)
                COFF = np.array(COFF)
                                
                FEATURE = ['Speed', 'Acce', 'Weight', 'PelletSelf', 'PelletOther', 'Interval','Entry']
                fig, axs = plt.subplots(1,7,figsize = (35,4))
                for j in range(len(FEATURE)):
                        axs[j].hist(COFF.T[j], color = 'blue', bins = 20, density = True, label = 'std='+str(round(np.std(COFF.T[j]), 3)))
                        axs[j].axvline(coefficients[j], color = 'red', linestyle = '--', label = 'coef='+str(round(coefficients[j], 3)))
                        axs[j].legend()
                        axs[j].set_xlabel(FEATURE[j])
                plt.savefig('/nfs/nhome/live/zimol/ProjectAeon/aeon_mecha/images/Regression/'+title+'Parameters.png')

if __name__ == "__main__":
        main()