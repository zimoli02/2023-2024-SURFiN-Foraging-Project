import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)

import sys
from pathlib import Path

current_script_path = Path(__file__).resolve()
parent_dir = current_script_path.parent.parent
sys.path.insert(0, str(parent_dir))

import SSM.ssm as ssm
from SSM.ssm.util import find_permutation
from SSM.ssm.plots import gradient_cmap, white_to_color_cmap

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_style("white")
sns.set_context("talk")

color_names = ["blue","red","yellow", "green","brown"]
colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)

def PlotLLS(hmm_lls):
    plt.plot(hmm_lls, label="EM")
    #plt.plot([0, N_iters], true_ll * np.ones(2), ':k', label="True")
    plt.xlabel("EM Iteration")
    plt.ylabel("Log Probability")
    plt.legend(loc="lower right")
    plt.show()



def PlotStates(hmm_z, mouse_pos, N):
    color_names = ["blue","red","yellow", "green","brown","purple","orange", "black"]
    colors = sns.xkcd_palette(color_names[0:N])
    cmap = gradient_cmap(colors)

    times = pd.to_datetime(mouse_pos.index)
    numerical_times = (times - times[0]).total_seconds().values
    states_array = hmm_z.reshape(1, -1)
    
    fig, axs = plt.subplots(1, 1, figsize=(35, 4))
    extent = [numerical_times[0], numerical_times[-1], 0, 1]
    cax = axs.imshow(states_array, aspect="auto", cmap=cmap, vmin=0, vmax=N-1, extent=extent)
    
    axs.set_xlabel('Time')
    axs.set_xticks(numerical_times[::len(numerical_times)//10])
    axs.set_xticklabels([time.strftime('%H:%M:%S') for time in times[::len(times)//10]], rotation=45, ha='right')
    
    axs.set_ylabel("States")
    axs.set_yticks([])

    cbar = fig.colorbar(cax, ax=axs, orientation='vertical')
    cbar.set_ticks(np.arange(0, N))
    cbar.set_ticklabels([f'State {val}' for val in np.arange(0, N)])
    
    plt.tight_layout()
    plt.show()


def PlotTransition(transition_mat):
    annot_array = np.array([[round(item, 2) for item in row] for row in transition_mat])
    fig, axs = plt.subplots(1,1, figsize=(4, 4))
    sns.heatmap(transition_mat, cmap='YlGnBu', ax = axs, square = 'True', cbar = False, annot=annot_array)
    axs.set_title("Learned Transition Matrix")
    plt.show()  



def FitHMM(data, num_states, n_iters = 50):
    obs_dim = len(data[0])
    hmm = ssm.HMM(num_states, obs_dim, observations="gaussian")
    
    lls = hmm.fit(data, method="em", num_iters=n_iters, init_method="kmeans")
    states = hmm.most_likely_states(data)
    transition_mat = hmm.transitions.transition_matrix
    
    return hmm, states, transition_mat, lls