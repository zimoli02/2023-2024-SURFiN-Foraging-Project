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

    # Filter data
    a, p, F, A, P = lds.ApplyFilter(a1, p1, e, n, y)
    v = [y[i] - a[i] for i in range(len(y))]

    # Smooth data
    A_, P_, r, N = lds.ApplySmoothing(a, p, F, y, v, e)
    t, end = 0,100
    years_xlabel = [1880, 1900, 1920, 1940, 1960]
    upper_bound = [A_[i] + 1.65*(P_[i]**0.5) for i in range(len(y))]
    lower_bound = [A_[i] - 1.65*(P_[i]**0.5) for i in range(len(y))]
    
    fig, axs = plt.subplots(2,2, figsize = (20,14))
    axs[0,0].scatter(years, y, color = 'black')
    axs[0,0].plot(years[t:end], A_[t:end], color = 'black', alpha = 0.7)
    axs[0,0].plot(years[t:end], upper_bound[t:end], color = 'grey', alpha = 0.7)
    axs[0,0].plot(years[t:end], lower_bound[t:end], color = 'grey', alpha = 0.7)
    axs[0,0].set_yticks([500, 750, 1000, 1250, 1500])
    
    axs[0,1].plot(years[t:end], P_[t:end], color = 'black')
    axs[0,1].set_yticks([2500, 3000, 3500, 4000])
    
    t, end = 0,-1
    zeroline = np.zeros(timepoints)
    axs[1,0].plot(years[t:end], r[t:end], color = 'black')
    axs[1,0].plot(years[t:end], zeroline[t:end], color = 'grey')
    axs[1,0].set_yticks([-0.02, 0, 0.02])

    axs[1,1].plot(years[t:end], N[t:end], color = 'black')
    axs[1,1].set_yticks([6e-5, 8e-5, 0.0001])

    for i in range(2):
        for j in range(2):
            axs[i,j].spines['right'].set_visible(False)
            axs[i,j].spines['top'].set_visible(False)
            axs[i,j].set_xticks(years_xlabel)
    
    fig.suptitle("Nile data and output of state smoothing recursion", y = 0.0)
    plt.savefig(figures_dir / 'Fig_2_2.png')

if __name__ == "__main__":
    main()
