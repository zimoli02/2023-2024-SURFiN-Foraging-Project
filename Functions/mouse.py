import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from collections import Counter
import h5py
from datetime import datetime, timedelta
import torch


import sys
from pathlib import Path

current_script_path = Path(__file__).resolve()
function_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(function_dir))
import Functions.learning as learning
import Functions.inference as inference
import SSM.ssm as ssm
#import Functions.HMM as HMM
import Functions.kinematics as kinematics
import Functions.patch as patch
from SSM.ssm.plots import gradient_cmap

parent_dir = current_script_path.parents[2] / 'aeon_mecha' 
sys.path.insert(0, str(parent_dir))
import aeon
import aeon.io.api as api
from aeon.schema.schemas import social02

nodes_name = ['nose', 'head', 'right_ear', 'left_ear', 'spine1', 'spine2','spine3', 'spine4']

class HMM:
    def __init__(self, mouse, n_state = None, feature = None):
        self.mouse = mouse
        self.n_state = n_state
        self.feature = feature
        self.features = None
        self.model = None
        self.model_period = self.mouse.active_chunk
        self.parameters = None
        self.states = None
        self.TransM = None
        self.loglikelihood = None
        
    def Get_Features(self):
        if self.feature == 'Kinematics': self.features = ['smoothed_speed', 'smoothed_acceleration', 'r']
        if self.feature == 'Body': self.features = ['bodylength', 'bodyangle', 'nose']
        if self.feature == 'Kinematics_and_Body': self.features = ['smoothed_speed', 'smoothed_acceleration', 'r', 'bodylength', 'bodyangle', 'nose']
    
    def Fit_Model(self):
        self.Get_Features()
        fitting_input = self.mouse.mouse_pos[self.model_period[0]:self.model_period[1]][self.features]
        self.model = ssm.HMM(self.n_state, len(fitting_input[0]), observations="gaussian")
        self.parameters = self.model.observations.params[0].T
        state_mean_speed = self.parameters[0]
        index = np.argsort(state_mean_speed, -1) 
        self.TransM = self.model.transitions.transition_matrix[index].T[index].T
        np.save('../SocialData/HMMStates/' + self.mouse.type + "_" + self.mouse.mouse + 'TransM.npy', self.TransM)
    
    def Fit_Model_without_Saving(self):
        self.Get_Features()
        fitting_input = self.mouse.mouse_pos[self.model_period[0]:self.model_period[1]][self.features]
        obs = self.mouse.mouse_pos[self.features]
        
        self.model = ssm.HMM(self.n_state, len(fitting_input[0]), observations="gaussian")
        self.parameters = self.model.observations.params[0].T
        self.states = self.model.most_likely_states(obs)
        self.loglikelihood = self.model.fit(obs, method="em", num_iters=self.n_state, init_method="kmeans")
        
        state_mean_speed = self.parameters[0]
        index = np.argsort(state_mean_speed, -1) 
        new_values = np.empty_like(self.states)
        for i, val in enumerate(index): new_values[self.states == val] = i
        self.states = new_values
        self.TransM = self.model.transitions.transition_matrix[index].T[index].T
    
    def Get_TransM(self):
        try:
            self.TransM = np.load('../SocialData/HMMStates/' + self.mouse.type + "_" + self.mouse.mouse + "TransM.npy", allow_pickle=True)
        except FileNotFoundError:
            self.Fit_Model()
    
    def Get_States(self):
        try:
            self.states = np.load('../SocialData/HMMStates/' + self.mouse.type + "_" + self.mouse.mouse + "_States.npy", allow_pickle=True)
        except FileNotFoundError:
            if self.model == None: self.Fit_Model()
            obs = self.mouse.mouse_pos[self.features]
            self.states = self.model.most_likely_states(obs)
            self.loglikelihood = self.model.fit(obs, method="em", num_iters=self.n_state, init_method="kmeans")

            state_mean_speed = self.parameters[0]
            index = np.argsort(state_mean_speed, -1)     

            new_values = np.empty_like(self.states)
            for i, val in enumerate(index): new_values[self.states == val] = i
            self.states = new_values
            np.save('../SocialData/HMMStates/' + self.mouse.type + "_" + self.mouse.mouse + "_States.npy", self.states)
        
        
class Arena:
    def __init__(self, mouse, origin = [738.7019332885742, 562.5901412251667], radius = 468.9626404164694):
        self.mouse = mouse
        self.root = self.mouse.root
        self.start = self.mouse.mouse_pos.index[0]
        self.end = self.mouse.mouse_pos.index[-1]
        self.origin = origin
        self.radius = radius
        self.entry = self.Entry()
        self.patch = ['Patch1', 'Patch2', 'Patch3']

        self.move_wheel_interval_seconds = 10
        self.pre_visit_seconds = 10
        
        self.pellets_per_patch = None
        self.pellets = None
        self.visits = None
    
    def Entry(self):
        distance = np.sqrt((self.mouse.mouse_pos['smoothed_position_x'] - self.origin[0]) ** 2 + (self.mouse.mouse_pos['smoothed_position_y'] - self.origin[1]) ** 2)
        return self.mouse.mouse_pos.iloc[np.where(distance < self.radius)].index
    
    def Get_Pellets_per_Patch(self):
        social02.Patch1.DeliverPellet.value = 1
        pellets1 = aeon.load(self.root, social02.Patch1.DeliverPellet, start=self.start, end=self.end)

        social02.Patch2.DeliverPellet.value = 1
        pellets2 = aeon.load(self.root, social02.Patch2.DeliverPellet, start=self.start, end=self.end)

        social02.Patch3.DeliverPellet.value = 1
        pellets3 = aeon.load(self.root, social02.Patch3.DeliverPellet, start=self.start, end=self.end)
            
        self.pellets_per_patch = {'Patch1':pellets1, 'Patch2':pellets2, 'Patch3':pellets3}
    def Get_Pellets(self):
        try:
            PELLET = pd.read_parquet('../SocialData/Pellets/'+ self.mouse.type + "_" + self.mouse.mouse +'_PELLET.parquet', engine='pyarrow')
        except FileNotFoundError:
            if self.pellets_per_patch == None: self.Get_Pellets_per_Patch()
            
            PELLET = pd.concat([self.pellets_per_patch['Patch1'],self.pellets_per_patch['Patch2'], self.pellets_per_patch['Patch3']], ignore_index=False)
            PELLET = PELLET.sort_index()
            PELLET.to_parquet('../SocialData/Pellets/'+ self.mouse.type + "_" + self.mouse.mouse +'_PELLET.parquet', engine='pyarrow')
        self.pellets = PELLET
    
    def Move_Wheel(self, start, end, patch):
        interval_seconds = move_wheel_interval_seconds
        starts, ends = [],[]
        while start < end:
            if start.minute != 0:
                end_ = pd.Timestamp(year = start.year, month = start.month, day = start.day, hour = start.hour+1, minute=0, second=0) - pd.Timedelta('2S')
            else:
                end_ = start + pd.Timedelta('1H') - pd.Timedelta('2S')
            starts.append(start+ pd.Timedelta('1S'))
            ends.append(end_)
            start = end_ + pd.Timedelta('2S')

        encoders = []
        if patch == 'Patch1': 
            for i in range(len(starts)):
                start, end = starts[i], ends[i]
                encoder = aeon.load(self.root, social02.Patch1.Encoder, start=start, end=end)
                encoders.append(encoder)
        elif patch == 'Patch2': 
            for i in range(len(starts)):
                start, end = starts[i], ends[i]
                encoder = aeon.load(self.root, social02.Patch2.Encoder, start=start, end=end)
                encoders.append(encoder)
        else: 
            for i in range(len(starts)):
                start, end = starts[i], ends[i]
                encoder = aeon.load(self.root, social02.Patch3.Encoder, start=start, end=end)
                encoders.append(encoder)
        encoder = pd.concat(encoders, ignore_index=False)
        encoder = encoder[::5]
        
        encoder = encoder.sort_index()
        encoder = encoder[~encoder.index.duplicated(keep='first')]
        
        w = -distancetravelled(encoder.angle).to_numpy()
        dw = np.concatenate((np.array([0]), w[:-1]- w[1:]))
        encoder['Distance'] = pd.Series(w, index=encoder.index)
        encoder['DistanceChange'] = pd.Series(dw, index=encoder.index)
        encoder['DistanceChange'] = encoder['DistanceChange'].rolling('10S').mean()
        encoder['Move'] = np.where(abs(encoder.DistanceChange) > 0.001, 1, 0)
        
        if interval_seconds < 0.01: return encoder
        groups = encoder['Move'].ne(encoder['Move'].shift()).cumsum()
        one_groups = encoder[encoder['Move'] == 1].groupby(groups).groups
        one_groups = list(one_groups.values())

        for i in range(len(one_groups) - 1):
            end_current_group = one_groups[i][-1]
            start_next_group = one_groups[i + 1][0]
            duration = start_next_group - end_current_group

            if duration < pd.Timedelta(seconds=interval_seconds):
                encoder.loc[end_current_group:start_next_group, 'Move'] = 1
                
        return encoder

    def Visits_in_Patch(self, patch):
        visits = []
        for i in range(len(self.mouse.starts)):
            if i == len(self.mouse.starts) - 1: mouse_pos_ = self.mouse.mouse_pos[self.mouse.starts[i]:]
            else: mouse_pos_ = mouse_pos[self.mouse.starts[i]:self.mouse.starts[i+1]]
            start, end = mouse_pos_.index[0], mouse_pos_.index[-1]
            encoder = self.Move_Wheel(start, end, patch = patch)
        
            visit = {'start':[],'end':[], 'duration':[], 'speed':[], 'acceleration':[], 'bodylength':[], 'bodyangle':[], 'nose' : [],'entry':[], 'patch':[], 'pellet':[]}
            
            groups = encoder['Move'].ne(encoder['Move'].shift()).cumsum()
            moves = encoder[encoder['Move'] == 1].groupby(groups)['Move']
            for name, group in moves:
                visit_start, visit_end = group.index[0], group.index[-1] - pd.Timedelta('10S')
                if (visit_end-visit_start).total_seconds() < 0: continue
                visit['start'].append(visit_start)
                visit['end'].append(visit_end)
                visit['duration'].append((visit_end-visit_start).total_seconds())
                #Visits['distance'].append(encoder.loc[start, 'Distance']-encoder.loc[end, 'Distance'])
                    
                pre_end = visit_start
                pre_start = pre_end - pd.Timedelta(seconds = self.pre_visit_seconds)
                if pre_start < mouse_pos_.index[0]: pre_start = mouse_pos_.index[0]
                
                index = self.entry.searchsorted(pre_end, side='left') - 1
                index = max(index, 0)
                visit['entry'].append((pre_end - self.entry[index]).total_seconds())
                    
                pre_visit_data = mouse_pos_.loc[pre_start:pre_end]
                
                visit['speed'].append(pre_visit_data['smoothed_speed'].mean())
                visit['acceleration'].append(pre_visit_data['smoothed_acceleration'].mean())
                visit['bodylength'].append(pre_visit_data['bodylength'].mean())
                visit['bodyangle'].append(pre_visit_data['bodyangle'].mean())
                visit['nose'].append(pre_visit_data['nose'].mean())
                
                #Visits['weight'].append(pre_visit_data['weight'].mean())
                #Visits['state'].append(pre_visit_data['states'].value_counts().idxmax())
                
                visit['patch'].append(patch)
                visit['pellet'].append(len(self.pellets_per_patch[patch]))
            
            visits.append(visit)
        visits = pd.concat(visits, ignore_index=True)
        return visits
        
    def Combine_Visits_in_Patch(self, Visits):
        Visits = pd.concat(Visits, ignore_index=True)

        Visits = Visits[Visits['duration'] >= 5]
        Visits = Visits.sort_values(by='start',ignore_index=True)  
        
        Visits['last_pellets_self'] = 0
        Visits['last_pellets_other'] = 0
        Visits['last_interval'] = 0
        Visits['next_interval'] = 0
        Visits['last_duration'] = 0
        Visits['last_pellets_interval'] = 0
        
        for i in range(1,len(Visits)):
            start, end = Visits.start[i], Visits.end[i]
            last_end = Visits.end[i-1]
            Visits.loc[i, 'last_interval'] = (start - last_end).total_seconds()
            Visits.loc[i-1, 'next_interval'] = Visits.loc[i, 'last_interval'] 
            Visits.loc[i, 'last_duration'] = Visits.loc[i-1, 'duration']
            
            self_patch, other_patch = False, False
            self_pellet, other_pellet = 0, 0
            for j in range(i-1, -1, -1):
                if self_patch and other_patch: break
                if Visits.patch[j] != Visits.patch[i] and other_patch == False: 
                    other_pellet = Visits.pellet[j]
                    other_patch = True
                if Visits.patch[j] == Visits.patch[i] and self_patch == False:
                    self_pellet = Visits.pellet[j]
                    self_patch = True
            Visits.loc[i, 'last_pellets_self'] = self_pellet
            Visits.loc[i, 'last_pellets_other'] = other_pellet
            
            if Visits.loc[i-1, 'pellet'] > 0:
                Visits.loc[i, 'last_pellets_interval'] = Visits.loc[i, 'last_interval']
            else:
                Visits.loc[i, 'last_pellets_interval'] = Visits.loc[i, 'last_interval'] + Visits.loc[i-1, 'last_pellets_interval'] + Visits.loc[i-1, 'duration']
        
        Visits = Visits.dropna(subset=['speed'])
        Visits = Visits[Visits['last_interval'] >= 0]
        Visits = Visits[Visits['next_interval'] >= 0]
            
        return Visits
        
    def Get_Visits(self):
        try:
            Visits = pd.read_parquet('../SocialData/VisitData/'  + self.mouse.type + "_" + self.mouse.mouse +'_Visit.parquet', engine='pyarrow')
        except FileNotFoundError:
            if self.pellets_per_patch == None: self.Get_Pellets()
            Visits = []
            for patch in self.patch:
                visits = self.Visits_in_Patch(patch)
                print('Visits for ' + patch + " Completed")
                Visits.append(visits)
            Visits = self.Combine_Visits_in_Patch(Visits)
            Visits = Visits.sort_values(by='start',ignore_index=True) 
            Visits.to_parquet('../SocialData/VisitData/'  + self.mouse.type + "_" + self.mouse.mouse +'_Visit.parquet', engine='pyarrow')
        self.visits = Visits
        

    
    
    
    

class Kinematics:
    def __init__(self, session):
        self.session = session
        self.inference_chunk = np.array([9, 10])
        self.manual_parameters = self.Get_Manual_Parameters()
        self.parameters = {}
        self.filterRes = None
        self.smoothRes = None
    def Run(self):
        self.Infer_Parameters()
        self.Inference() 
        
    def Get_Manual_Parameters(self):
        dt = 0.1
        pos_x0, pos_y0 = 0, 0
        vel_x0, vel_y0 = 0.0, 0.0
        acc_x0, acc_y0 = 0.0, 0.0

        # Manual Parameters
        sigma_a = 1.3
        sqrt_diag_V0_value = 1e-3

        m0 = np.array([pos_x0, vel_x0, acc_x0, pos_y0, vel_y0, acc_y0], dtype=np.double)
        V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)


        B = np.array([[1, dt, dt**2/2, 0, 0, 0],
                    [0, 1, dt, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, dt, dt**2/2],
                    [0, 0, 0, 0, 1, dt],
                    [0, 0, 0, 0, 0, 1]],
                    dtype=np.double)


        Qe = np.array([[dt**4/4, dt**3/2, dt**2/2, 0, 0, 0],
                    [dt**3/2, dt**2,   dt,      0, 0, 0],
                    [dt**2/2, dt,      1,       0, 0, 0],
                    [0, 0, 0, dt**4/4, dt**3/2, dt**2/2],
                    [0, 0, 0, dt**3/2, dt**2,   dt],
                    [0, 0, 0, dt**2/2, dt,      1]],
                    dtype=np.double)
        Q = sigma_a**2 * Qe

        sigma_x = 1
        sigma_y = 1

        Z = np.array([[1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0]],
                    dtype=np.double)
        R = np.diag([sigma_x**2, sigma_y**2])
        
        parameters = {'sigma_a': sigma_a,
                    'sigma_x': sigma_x,
                    'sigma_y': sigma_y,
                    'sqrt_diag_V0_value': sqrt_diag_V0_value,
                    'B': B,
                    'Qe': Qe,
                    'm0': m0,
                    'V0': V0,
                    'Z': Z,
                    'R': R}
        return parameters
    
    def Learn_Parameters(self, y, sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, Z):
        # Learning Parameters
        lbfgs_max_iter = 2
        lbfgs_tolerance_grad = 1e-3
        lbfgs_tolerance_change = 1e-3
        lbfgs_lr = 1.0
        lbfgs_n_epochs = 100
        lbfgs_tol = 1e-3
        
        Qe_reg_param_learned = 1e-10
        sqrt_diag_R_torch = torch.DoubleTensor([sigma_x, sigma_y])
        m0_torch = torch.from_numpy(m0.copy())
        sqrt_diag_V0_torch = torch.DoubleTensor([sqrt_diag_V0_value
                                                for i in range(len(m0))])
        if Qe_reg_param_learned is not None:
            Qe_regularized_learned = Qe + Qe_reg_param_learned * np.eye(Qe.shape[0])
        else:
            Qe_regularized_learned = Qe
        y_torch = torch.from_numpy(y.astype(np.double))
        B_torch = torch.from_numpy(B.astype(np.double))
        Qe_regularized_learned_torch = torch.from_numpy(Qe_regularized_learned.astype(np.double))
        Z_torch = torch.from_numpy(Z.astype(np.double))

        vars_to_estimate = {}
        vars_to_estimate["sigma_a"] = True
        vars_to_estimate["sqrt_diag_R"] = True
        vars_to_estimate["R"] = True
        vars_to_estimate["m0"] = True
        vars_to_estimate["sqrt_diag_V0"] = True
        vars_to_estimate["V0"] = True

        optim_res_learned = learning.torch_lbfgs_optimize_SS_tracking_diagV0(
            y=y_torch, B=B_torch, sigma_a0=sigma_a,
            Qe=Qe_regularized_learned_torch, Z=Z_torch, sqrt_diag_R_0=sqrt_diag_R_torch, m0_0=m0_torch,
            sqrt_diag_V0_0=sqrt_diag_V0_torch, max_iter=lbfgs_max_iter, lr=lbfgs_lr,
            vars_to_estimate=vars_to_estimate, tolerance_grad=lbfgs_tolerance_grad,
            tolerance_change=lbfgs_tolerance_change, n_epochs=lbfgs_n_epochs,
            tol=lbfgs_tol)
        
        sigma_a = optim_res_learned["estimates"]["sigma_a"].item()
        sigma_x = optim_res_learned["estimates"]["sqrt_diag_R"].numpy()[0]
        sigma_y = optim_res_learned["estimates"]["sqrt_diag_R"].numpy()[1]
        sqrt_diag_V0_value = optim_res_learned["estimates"]["sqrt_diag_V0"].numpy()
        m0 = optim_res_learned["estimates"]["m0"].numpy()
        V0 = np.diag(sqrt_diag_V0_value**2)
        R = np.diag(optim_res_learned["estimates"]["sqrt_diag_R"].numpy()**2)

        return sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value[0], B, m0, V0, Z, R
    
    def Infer_Parameters(self):
        try:
            P = np.load('../SocialData/LDS_Parameters/' + self.session.start + '_Parameters.npz', allow_pickle=True)
            sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, V0, Z, R = P['sigma_a'].item(), P['sigma_x'].item(), P['sigma_y'].item(), P['sqrt_diag_V0_value'].item(), P['B'], P['Qe'], P['m0'], P['V0'], P['Z'], P['R']
        except FileNotFoundError:
            #10Hz Data
            obs = np.transpose(self.session.mouse_pos[["x", "y"]].to_numpy())[:, :20*60*10]
            
            params = self.manual_parameters
            sigma_a = params['sigma_a']
            sigma_x = params['sigma_x']
            sigma_y = params['sigma_y']
            sqrt_diag_V0_value = params['sqrt_diag_V0_value']
            B = params['B']
            Qe = params['Qe']
            m0 = params['m0']
            V0 = params['V0']
            Z = params['Z']
            R = params['R']

            sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, m0, V0, Z, R = self.Learn_Parameters(obs, sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, Z)
            np.savez('../SocialData/LDS_Parameters/' + self.session.start + '_Parameters.npz', sigma_a = sigma_a, sigma_x = sigma_x, sigma_y = sigma_y, sqrt_diag_V0_value = sqrt_diag_V0_value, B = B, Qe = Qe, m0 = m0, V0 = V0, Z = Z, R = R)
            print('Inferring LDS Parameters Completed')
        
        parameters = {'sigma_a': sigma_a,
                    'sigma_x': sigma_x,
                    'sigma_y': sigma_y,
                    'sqrt_diag_V0_value': sqrt_diag_V0_value,
                    'B': B,
                    'Qe': Qe,
                    'm0': m0,
                    'V0': V0,
                    'Z': Z,
                    'R': R}

        self.parameters = parameters
    
    def Inference(self):
        try:
            filterRes = np.load('../SocialData/LDS/' + self.session.start +'_filterRes.npz')
            smoothRes = np.load('../SocialData/LDS/' + self.session.start +'_smoothRes.npz')
        except FileNotFoundError:
            obs = np.transpose(self.session.mouse_pos[["x", "y"]].to_numpy())
            
            params = self.parameters
            sigma_a = params['sigma_a']
            B = params['B']
            Qe = params['Qe']
            m0 = params['m0']
            V0 = params['V0']
            Z = params['Z']
            R = params['R']

            Q = (sigma_a**2) * Qe

            # Filtering
            filterRes = inference.filterLDS_SS_withMissingValues_np(
                y=obs, B=B, Q=Q, m0=m0, V0=V0, Z=Z, R=R)
            np.savez_compressed('../SocialData/LDS/' + self.session.start +'_filterRes.npz', **filterRes)
                
            # Smoothing
            smoothRes = inference.smoothLDS_SS( 
                B=B, xnn=filterRes["xnn"], Vnn=filterRes["Vnn"],
                xnn1=filterRes["xnn1"], Vnn1=filterRes["Vnn1"], m0=m0, V0=V0)
            np.savez_compressed('../SocialData/LDS/' + self.session.start +'_smoothRes.npz', **smoothRes) 
            
        self.filterRes = filterRes
        self.smoothRes = smoothRes

class Body_Info:
    def __init__(self, mouse):
        self.mouse = mouse
        self.bodylength = None
        self.bodyangle = None
        self.nose = None
    def Process_Body_Length(self): 
        dx = self.mouse.body_data_x['spine4'] - self.mouse.body_data_x['head']
        dy = self.mouse.body_data_y['spine4'] - self.mouse.body_data_y['head']
        self.bodylength = np.sqrt(dx**2 + dy**2)
    def Process_Body_Angle(self): 
        head = np.array([self.mouse.body_data_x['head'], self.mouse.body_data_y['head']]).T
        spine2 = np.array([self.mouse.body_data_x['spine2'], self.mouse.body_data_y['spine2']]).T
        spine4 = np.array([self.mouse.body_data_x['spine4'], self.mouse.body_data_y['spine4']]).T
        v1 = head - spine2
        v2 = spine4 - spine2
        dot_product = np.einsum('ij,ij->i', v1, v2)
        norm1 = np.linalg.norm(v1, axis=1)
        norm2 = np.linalg.norm(v2, axis=1)
        cos_theta = dot_product / (norm1 * norm2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radians_theta = np.arccos(cos_theta)
        self.bodyangle = np.degrees(radians_theta)
    def Process_Nose(self):
        mid_x = (self.mouse.body_data_x['right_ear'] + self.mouse.body_data_x['left_ear'])/2
        mid_y = (self.mouse.body_data_y['right_ear'] + self.mouse.body_data_y['left_ear'])/2
        dx = self.mouse.body_data_x['nose'] - mid_x
        dy = self.mouse.body_data_y['nose'] - mid_y
        self.nose = np.sqrt(dx**2 + dy**2)
    def Run(self):
        self.Process_Body_Length()
        self.Process_Body_Angle()
        self.Process_Nose()

class Mouse:
    def __init__(self, aeon_exp, type, mouse):
        self.aeon_exp = aeon_exp
        self.type = type
        self.mouse = mouse
        self.root = self.Get_Root()
        self.INFO = self.Get_INFO()
        self.starts, self.ends = self.Get_Start_Times()
        self.body_data_x, self.body_data_y = self.Combine_SLEAP_Data()
        self.mouse_pos = self.Get_Mouse_Pos()
        self.active_chunk = self.Get_Active_Chunk()
        
        self.arena = Arena(self)
        self.body_info = Body_Info(self)
        self.hmm = HMM(self)

        
    
    def Get_Root(self):
        if self.aeon_exp == 'AEON3': return '/ceph/aeon/aeon/data/raw/AEON3/social0.2/'
        if self.aeon_exp == 'AEON4': return '/ceph/aeon/aeon/data/raw/AEON4/social0.2/'
        
    def Get_INFO(self):
        experiment = Experiment()
        if self.aeon_exp == 'AEON3': return experiment.INFO3
        if self.aeon_exp == 'AEON4': return experiment.INFO4
        
    def Get_Start_Times(self):
        starts, ends = [], []
        for i in range(len(self.INFO["Mouse"])):
            if self.INFO['Type'][i] == self.type and self.INFO["Mouse"][i] == self.mouse:
                starts.append(self.INFO["Start"][i])
                ends.append(self.INFO["End"][i])
        return starts, ends
    
    def Combine_SLEAP_Data(self):
        try:
            data_x = pd.read_parquet('../SocialData/BodyData/' + self.type + "_" + self.mouse + '_x.parquet', engine='pyarrow')
            data_y = pd.read_parquet('../SocialData/BodyData/' + self.type + "_" + self.mouse + '_y.parquet', engine='pyarrow')
        except FileNotFoundError:
            data_x, data_y= [], []
            for i in range(len(self.starts)):
                session = Session(aeon_exp = self.aeon_exp, type = self.type, mouse = self.mouse, start = self.starts[i], end = self.ends[i])
                x, y = session.body_data_x, session.body_data_y
                data_x.append(x)
                data_y.append(y)
                
            data_x = pd.concat(data_x, ignore_index=False)
            data_y = pd.concat(data_y, ignore_index=False)
                
            data_x.to_parquet('../SocialData/BodyData/' + self.type + "_" + self.mouse + '_x.parquet', engine='pyarrow')
            data_y.to_parquet('../SocialData/BodyData/' + self.type + "_" + self.mouse + '_y.parquet', engine='pyarrow')
        
        return data_x, data_y
        
    def Get_Mouse_Pos(self):
        mouse_pos = []
        for i in range(len(self.starts)):
            session = Session(aeon_exp = self.aeon_exp, type = self.type, mouse = self.mouse, start = self.starts[i], end = self.ends[i])
            session.Add_Kinematics()
            mouse_pos.append(session.mouse_pos)
        mouse_pos = pd.concat(mouse_pos, ignore_index=False)
        return mouse_pos
    
    def Get_Active_Chunk(self):
        start = self.mouse_pos.index[0]
        try:
            start_ = pd.Timestamp(year = start.year, month = start.month, day = start.day+1, hour = 7, minute=0, second=0)
        except ValueError:
            start_ = pd.Timestamp(year = start.year, month = start.month+1, day = 1, hour = 7, minute=0, second=0)
        try:
            end_ = pd.Timestamp(year = start_.year, month = start_.month, day = start_.day+1, hour = 7, minute=0, second=0)
        except ValueError:
            end_ = pd.Timestamp(year = start_.year, month = start_.month+1, day = 1, hour = 7, minute=0, second=0)
        return np.array([start_,end_])
    
    def FixNan(self, mouse_pos, column):
        mouse_pos[column] = mouse_pos[column].interpolate()
        mouse_pos[column] = mouse_pos[column].ffill()
        mouse_pos[column] = mouse_pos[column].bfill()
        return mouse_pos 

    def Add_Features_to_mouse_pos(self):
        self.Add_Body_Info_to_mouse_pos()
        self.Add_Distance_to_mouse_pos()    

    def Add_Body_Info_to_mouse_pos(self):
        self.body_info.Run()
        self.mouse_pos['bodylength'] = pd.Series(self.body_info.bodylength, index=self.mouse_pos.index)
        self.mouse_pos = self.FixNan(self.mouse_pos,'bodylength')    
        self.mouse_pos['bodyangle'] = pd.Series(self.body_info.bodyangle, index=self.mouse_pos.index)
        self.mouse_pos = self.FixNan(self.mouse_pos,'bodyangle')    
        self.mouse_pos['nose'] = pd.Series(self.body_info.nose, index=self.mouse_pos.index)
        self.mouse_pos = self.FixNan(self.mouse_pos,'nose')    
        
    def Add_Distance_to_mouse_pos(self):
        distance = np.sqrt((self.mouse_pos['smoothed_position_x'] - self.arena.origin[0]) ** 2 + (self.mouse_pos['smoothed_position_y'] - self.arena.origin[1]) ** 2)
        self.mouse_pos['r'] = distance
    
    def Run_Visits(self):
        self.arena.Get_Pellets()
        self.arena.Get_Visits()
        
    
    
class Session:
    def __init__(self, aeon_exp, type, mouse, start, end):
        self.aeon_exp = aeon_exp
        self.type = type
        self.mouse = mouse
        self.start = start
        self.end = end
        self.file_path = self.Get_Filepath()
        self.body_data_x, self.body_data_y = self.Extract_SLEAP_Data()
        self.mouse_pos = self.Get_Mouse_Pos()
        self.kinematics = Kinematics(self)
    
    def DeleteNan(self, df, df_):
        temp_df = df.dropna(subset=['spine2'])
        first_valid_index, last_valid_index = temp_df.index[0], temp_df.index[-1]
        df = df.loc[first_valid_index:last_valid_index]
        df_ = df_.loc[first_valid_index:last_valid_index]
        
        temp_df = df_.dropna(subset=['spine2'])
        first_valid_index, last_valid_index = temp_df.index[0], temp_df.index[-1]
        df = df.loc[first_valid_index:last_valid_index]
        df_ = df_.loc[first_valid_index:last_valid_index]
        
        return df, df_

    def Get_Filepath(self):
        if self.aeon_exp == 'AEON3': self.file_path = "/ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraTop/predictions_social02/AEON3/analyses/CameraTop_"
        else: self.file_path = "/ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraTop/predictions_social02/AEON4/analyses/CameraTop_"
        
    def Extract_SLEAP_Data(self):
        try:
            data_x = pd.read_parquet('../SocialData/RawData/' + self.start + '_x.parquet', engine='pyarrow')
            data_y = pd.read_parquet('../SocialData/RawData/' + self.start + '_y.parquet', engine='pyarrow')
        except FileNotFoundError:
            start_chunk, end_chunk = datetime.strptime(self.start, '%Y-%m-%dT%H-%M-%S'), datetime.strptime(self.end, '%Y-%m-%dT%H-%M-%S')
            chunk = start_chunk.replace(minute=0, second=0, microsecond=0)
            flag = 0
            chunks = []
            while flag == 0:
                chunk_time = chunk.strftime('%Y-%m-%dT%H-%M-%S')
                chunks.append(chunk_time)
                chunk += timedelta(hours=1)
                if chunk_time == self.end: flag = 1
            
            dfs_x, dfs_y = [], []
            for j in range(len(chunks)):
                file_path_ = self.file_path + chunks[j] + "_full_pose.analysis.h5"
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
            data_x, data_y = self.DeleteNan(data_x, data_y)
        
        return data_x, data_y

    def Get_Mouse_Pos(self):
        x = self.body_data_x['spine2']
        y = self.body_data_y['spine2']
        mouse_pos = pd.DataFrame({'x': x, 'y': y})
        mouse_pos['x'] = mouse_pos['x'].interpolate()
        mouse_pos['y'] = mouse_pos['x'].interpolate()
        return mouse_pos
    
    def Add_Kinematics(self):
        self.kinematics.Run()
        smoothRes = self.kinematics.smoothRes
        self.mouse_pos['smoothed_position_x'] = pd.Series(smoothRes['xnN'][0][0], index=self.mouse_pos.index)
        self.mouse_pos['smoothed_position_y'] = pd.Series(smoothRes['xnN'][3][0], index=self.mouse_pos.index)
    
        x_vel, y_vel = smoothRes['xnN'][1][0], smoothRes['xnN'][4][0]
        vel = np.sqrt(x_vel**2 + y_vel**2)
        self.mouse_pos['smoothed_speed'] = pd.Series(vel, index=self.mouse_pos.index)
            
        x_acc, y_acc = smoothRes['xnN'][2][0], smoothRes['xnN'][5][0]
        acc = np.sqrt(x_acc**2 + y_acc**2)
        self.mouse_pos['smoothed_acceleration'] = pd.Series(acc, index=self.mouse_pos.index)

class Experiment:
    def __init__(self):
        self.INFO3 =  { "Type": ["Pre", "Pre", "Pre", "Pre", "Post", "Post", "Post"],
                        "Mouse": ["BAA-1104045", "BAA-1104045", "BAA-1104045", "BAA-1104047","BAA-1104045", "BAA-1104047", "BAA-1104047"],
                        "Start": ["2024-01-31T11-28-39", "2024-02-01T22-36-47", "2024-02-02T00-15-00", "2024-02-05T15-43-07", "2024-02-25T17-22-33", "2024-02-28T13-54-17", "2024-03-01T16-46-12"],
                        "End": ["2024-02-01T20-00-00", "2024-02-01T23-00-00", "2024-02-03T16-00-00", "2024-02-08T14-00-00", "2024-02-28T12-00-00", "2024-03-01T15-00-00", "2024-03-02T15-00-00"]}
        self.INFO4 =  {"Type": ["Pre", "Pre", "Pre", "Pre", "Post", "Post"],
                        "Mouse": ["BAA-1104048", "BAA-1104048", "BAA-1104048", "BAA-1104049","BAA-1104048", "BAA-1104049"],
                        "Start": ["2024-01-31T10-14-14", "2024-02-01T20-46-44", "2024-02-01T23-40-29", "2024-02-05T14-36-00", "2024-02-25T17-24-32", "2024-02-28T13-45-06"],
                        "End": ["2024-02-01T19-00-00", "2024-02-01T22-00-00", "2024-02-03T16-00-00", "2024-02-08T14-00-00", "2024-02-28T12-00-00", "2024-03-02T15-00-00"]}




def main():
    print('None')
    
    
if __name__ == "__main__":
    main()