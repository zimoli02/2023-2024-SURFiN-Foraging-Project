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

def AddRegressors(Visits_Patch1, Visits_Patch2):
        Visits_Patch1 = Visits_Patch1.copy()
        Visits_Patch2 = Visits_Patch2.copy()

        Visits_Patch1['DistanceTravelledinVisit'] = 0
        for i in range(len(Visits_Patch1)):
                start, end = Visits_Patch1.start[i], Visits_Patch1.end[i]
                
                encoder1 = api.load(root, exp02.Patch1.Encoder, start=start, end=end)
                if encoder1.empty: Visits_Patch1.loc[i, 'DistanceTravelledinVisit'] = 0
                else:
                        w1 = -distancetravelled(encoder1.angle)
                        Visits_Patch1.loc[i, 'DistanceTravelledinVisit'] = w1[0]-w1[-1]
        
        Visits_Patch1['PelletsInLastVisit1'] = 0
        Visits_Patch1['PelletsInLastVisit2'] = 0
        for i in range(1,len(Visits_Patch1)):
                start, end = Visits_Patch1.start[i-1], Visits_Patch1.end[i-1]
                
                pellets_patch1 = api.load(root, exp02.Patch1.DeliverPellet, start=start, end=end)
                Visits_Patch1.loc[i, 'PelletsInLastVisit1'] = len(pellets_patch1)
                
                prior_timestamps = Visits_Patch2[Visits_Patch2['end'] < Visits_Patch1.start[i]]
                if prior_timestamps.empty: Visits_Patch1.loc[i, 'PelletsInLastVisit2'] = 0
                else:
                        start_, end_ = prior_timestamps.iloc[-1]['start'], prior_timestamps.iloc[-1]['end']
                        pellets_patch2 = api.load(root, exp02.Patch2.DeliverPellet, start=start_, end=end_)
                        Visits_Patch1.loc[i, 'PelletsInLastVisit2'] = len(pellets_patch2)
                
        Visits_Patch2['DistanceTravelledinVisit'] = 0
        for i in range(len(Visits_Patch2)):
                start, end = Visits_Patch2.start[i], Visits_Patch2.end[i]
                encoder2 = api.load(root, exp02.Patch2.Encoder, start=start, end=end)
                if encoder2.empty: Visits_Patch2.loc[i, 'DistanceTravelledinVisit'] = 0
                else:
                        w2 = -distancetravelled(encoder2.angle)
                        Visits_Patch2.loc[i, 'DistanceTravelledinVisit'] = w2[0]-w2[-1]
        
                
        Visits_Patch2['PelletsInLastVisit2'] = 0
        Visits_Patch2['PelletsInLastVisit1'] = 0
        for i in range(1,len(Visits_Patch2)):
                start, end = Visits_Patch2.start[i-1], Visits_Patch2.end[i-1]

                pellets_patch2 = api.load(root, exp02.Patch2.DeliverPellet, start=start, end=end)
                Visits_Patch2.loc[i, 'PelletsInLastVisit2'] = len(pellets_patch2)
                
                prior_timestamps = Visits_Patch1[Visits_Patch1['end'] < Visits_Patch2.start[i]]
                if prior_timestamps.empty: Visits_Patch2.loc[i, 'PelletsInLastVisit1'] = 0
                else:
                        start_, end_ = prior_timestamps.iloc[-1]['start'], prior_timestamps.iloc[-1]['end']
                        pellets_patch1 = api.load(root, exp02.Patch1.DeliverPellet, start=start_, end=end_)
                        Visits_Patch2.loc[i, 'PelletsInLastVisit1'] = len(pellets_patch1)

        return Visits_Patch1, Visits_Patch2


def main():
        subject_events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
        sessions = visits(subject_events[subject_events.id.str.startswith("BAA-")])
        short_sessions = sessions.iloc[[4,16,17,20,23,24,25,26,28,29,30,31]]
        
        
        for session, i in zip(list(short_sessions.itertuples()), range(len(short_sessions))):
                coefficients_1 = []
                coefficients_2 = []
                
                title = 'ShortSession'+str(i)
                print(title)

                mouse_pos = pd.read_parquet(title+'mousepos.parquet', engine='pyarrow')
                        
                Visits_Patch1 = patch.VisitPatch(mouse_pos, patch = 'Patch1', speed_ave_time = '5S', acce_ave_time = '5S', weight_ave_time = '5S')
                Visits_Patch2 = patch.VisitPatch(mouse_pos, patch='Patch2',speed_ave_time = '5S', acce_ave_time = '5S', weight_ave_time = '5S')
                Visits_Patch1, Visits_Patch2 = AddRegressors(Visits_Patch1, Visits_Patch2) 
                
                        
                X1 = Visits_Patch1[['speed','acceleration', 'weight','PelletsInLastVisit1', 'PelletsInLastVisit2']]
                y1 = Visits_Patch1['DistanceTravelledinVisit']
                X2 = Visits_Patch2[['speed','acceleration', 'weight','PelletsInLastVisit2', 'PelletsInLastVisit1']]
                y2 = Visits_Patch2['DistanceTravelledinVisit']
                
                scaler = StandardScaler()
                X1 = scaler.fit_transform(X1)
                scaler = StandardScaler()
                X2 = scaler.fit_transform(X2)
                        
                coefficient1, intercept1, y1_pred, mse1, r21 = patch.SimpleLinearRegression(X1, y1)
                coefficient2, intercept2, y2_pred, mse2, r22 = patch.SimpleLinearRegression(X2, y2)
                
                fig, axs = plt.subplots(4, 1, figsize=(30, 5), sharex=True)

                axs[0].plot(y1, color = 'green')
                axs[0].set_xticks([]) 
                axs[0].set_facecolor('white') 
                axs[0].set_ylabel('P1', fontsize = 12)
                axs[0].spines['top'].set_visible(False)
                axs[0].spines['right'].set_visible(False)

                axs[1].plot(y1_pred, color = 'green', label = 'r2='+str(round(r21, 3)))
                axs[1].set_xticks([]) 
                axs[1].set_facecolor('white') 
                axs[1].set_ylabel('P1 Pred.', fontsize = 14)
                axs[1].spines['top'].set_visible(False)
                axs[1].spines['right'].set_visible(False)
                axs[1].legend()

                axs[2].plot(y2, color = 'brown') 
                axs[2].set_facecolor('white')  
                axs[2].set_ylabel('P2', fontsize = 14)
                axs[2].spines['top'].set_visible(False)
                axs[2].spines['right'].set_visible(False)
                
                axs[3].plot(y2_pred, color = 'brown', label = 'r2='+str(round(r22, 3)))
                axs[3].set_xticks([]) 
                axs[3].set_facecolor('white') 
                axs[3].set_ylabel('P2 Pred.', fontsize = 12)
                axs[3].spines['top'].set_visible(False)
                axs[3].spines['right'].set_visible(False)
                axs[3].legend()
                
                plt.savefig('/nfs/nhome/live/zimol/ProjectAeon/aeon_mecha/images/Regression/'+title+'.png')
                        
                for t in range (1000):
                        index = np.random.choice(np.arange(len(y1)), size=len(y1), replace=True)
                        coefficients1, intercept1, y1_pred, mse1, r21 = patch.SimpleLinearRegression(X1[index], y1[index])
                        index = np.random.choice(np.arange(len(y2)), size=len(y2), replace=True)
                        coefficients2, intercept2, y2_pred, mse2, r22 = patch.SimpleLinearRegression(X2[index], y2[index])
                        coefficients_1.append(coefficients1)
                        coefficients_2.append(coefficients2)
                coefficients_1, coefficients_2 = np.array(coefficients_1), np.array(coefficients_2)
                
                FEATURE = ['Speed', 'Acce', 'Weight', 'PelletSelf', 'PelletOther']
                fig, axs = plt.subplots(2,5,figsize = (25,8))
                for j in range(5):
                        axs[0,j].hist(coefficients_1.T[j], color = 'blue', bins = 20, density = True, label = 'std='+str(round(np.std(coefficients_1.T[j]), 3)))
                        axs[0,j].axvline(coefficient1[j], color = 'red', linestyle = '--', label = 'coef='+str(round(coefficient1[j], 3)))
                        axs[0,j].legend()
                        axs[1,j].hist(coefficients_2.T[j], color = 'blue', bins = 20, density = True, label = 'std='+str(round(np.std(coefficients_2.T[j]), 3)))
                        axs[1,j].axvline(coefficient2[j], color = 'red', linestyle = '--', label = 'coef='+str(round(coefficient2[j], 3)))
                        axs[1,j].legend()
                        axs[1,j].set_xlabel(FEATURE[j])
                axs[0,0].set_ylabel('Patch 1')
                axs[1,0].set_ylabel('Patch 2')
                plt.savefig('/nfs/nhome/live/zimol/ProjectAeon/aeon_mecha/images/Regression/'+title+'_.png')

if __name__ == "__main__":
        main()