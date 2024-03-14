import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(parent_dir)
import Functions.lds as lds

def main():
    current_dir = Path(__file__).resolve().parent
    figures_dir = current_dir.parent / 'Figures'
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(current_dir,'Niles.csv'))
    years = data['years'].to_numpy()
    y = data['y'].to_numpy()

    a1, p1 = 0, 1e7
    e = 15099
    n = 1469.1
    timepoints = len(years)

    # Filter data: not using steady state
    a, p, F, A, P = lds.ApplyFilter(a1, p1, e, n, y)
    v = [y[i] - a[i] for i in range(len(y))]

    years_xlabel = [1880, 1900, 1920, 1940, 1960]
    upper_bound = [a[i] + 1.65*(p[i]**0.5) for i in range(len(y))]
    lower_bound = [a[i] - 1.65*(p[i]**0.5) for i in range(len(y))]
    
    fig, axs = plt.subplots(2,2, figsize = (18,14))
    axs[0,0].scatter(years, y, color = 'black')
    axs[0,0].plot(years[1:], a[1:], color = 'black', alpha = 0.7)
    axs[0,0].plot(years[1:], upper_bound[1:], color = 'grey', alpha = 0.7)
    axs[0,0].plot(years[1:], lower_bound[1:], color = 'grey', alpha = 0.7)
    axs[0,0].set_yticks([500, 750, 1000, 1250, 1500])
    
    axs[0,1].plot(years[1:], p[1:100], color = 'black') 
    axs[0,1].set_yticks([7500,10000, 12500, 15000, 17500])
    
    zeroline = np.zeros(timepoints)
    axs[1,0].plot(years[1:], v[1:], color = 'black')
    axs[1,0].plot(years, zeroline, color = 'grey')
    axs[1,0].set_yticks([-250, 0, 250])
    
    axs[1,1].plot(years[1:], F[1:], color = 'black')
    axs[1,1].set_yticks([22500, 25000, 27500, 30000, 32500])
    
    for i in range(2):
        for j in range(2):
            axs[i,j].spines['right'].set_visible(False)
            axs[i,j].spines['top'].set_visible(False)
            axs[i,j].set_xticks(years_xlabel)
    
    fig.suptitle("Nile data and output of Kalman filter", y = 0.0, fontsize = 25)
    plt.savefig(figures_dir / 'Fig_2_1.png')

    # Filter data: using steady_state when its difference with Kalman filter variance <= 1e-3 
    a_, p_, F_, A_, P_ = lds.ApplyFilter(a1, p1, e, n, y, steady_state=True)
    p_steadystate = lds.SteadyStateVariance(e,n)
    v_ = [y[i] - a_[i] for i in range(len(y))]

    upper_bound = [a_[i] + 1.65*(p_[i]**0.5) for i in range(len(y))]
    lower_bound = [a_[i] - 1.65*(p_[i]**0.5) for i in range(len(y))]
    
    fig, axs = plt.subplots(2,3, figsize = (28,14))
    axs[0,0].scatter(years, y, color = 'black')
    axs[0,0].plot(years[1:], a_[1:], color = 'black', alpha = 0.7)
    axs[0,0].plot(years[1:], upper_bound[1:], color = 'grey', alpha = 0.7)
    axs[0,0].plot(years[1:], lower_bound[1:], color = 'grey', alpha = 0.7)
    axs[0,0].set_yticks([500, 750, 1000, 1250, 1500])
    
    axs[0,1].plot(years[1:], p[1:100], color = 'black') 
    axs[0,1].plot(years[1:], p_[1:100], color = 'red') 
    axs[0,1].set_yticks([7500,10000, 12500, 15000, 17500])

    for i in range(1,len(p)):
        if abs(p[i] - p_steadystate) <= 0.001:
            n = i
            break
    axs[0,1].axvline(years[n], color = 'blue',linewidth = 1, linestyle = ':')

    dA = [np.sqrt((A[i] - A_[i])**2) for i in range(len(A))]
    axs[0,2].plot(years[1:], dA[1:], color = 'black')
    axs[0,2].set_ylabel("Mean-Squared Error")


    zeroline = np.zeros(timepoints)
    axs[1,0].plot(years[1:], v_[1:], color = 'black')
    axs[1,0].plot(years, zeroline, color = 'grey')
    axs[1,0].set_yticks([-250, 0, 250])
    
    axs[1,1].plot(years[1:], F[1:], color = 'black')
    axs[1,1].plot(years[1:], F_[1:], color = 'red')
    axs[1,1].set_yticks([22500, 25000, 27500, 30000, 32500])
    axs[1,1].axvline(years[n], color = 'blue',linewidth = 1, linestyle = ':')
    
    for i in range(2):
        for j in range(3):
            axs[i,j].spines['right'].set_visible(False)
            axs[i,j].spines['top'].set_visible(False)
            axs[i,j].set_xticks(years_xlabel)
    axs[1,2].set_xticks([])
    
    fig.suptitle("Nile data and output of Mixed Steady-State Kalman filter", y = 0.0, fontsize = 25)
    plt.savefig(figures_dir / 'Fig_2_1_SteadyState.png')
    
if __name__ == "__main__":
    main()
