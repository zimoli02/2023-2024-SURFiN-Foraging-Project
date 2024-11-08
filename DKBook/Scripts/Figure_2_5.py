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

    # Set Parameters
    a1, p1 = 0, 1e7
    e = 15099
    n = 1469.1
    timepoints = len(years)

    # Create Missing Observations
    y = np.array(y, dtype = float)
    for i in range(20,40): y[i] = np.nan
    for i in range(60,80): y[i] = np.nan

    # Apply Filtering
    a, p, F, A, P = lds.ApplyFilter(a1, p1, e, n, y)
    v = [y[i] - a[i] for i in range(len(y))]
    A_, P_, r, n = lds.ApplySmoothing(a, p, F, y, v, e)


    # Plotting
    years_xlabel = [1880, 1900, 1920, 1940, 1960]

    fig, axs = plt.subplots(2,2, figsize = (20,14))

    axs[0,0].plot(years, y, color = 'black')
    axs[0,0].plot(years, A, color = 'black', alpha = 0.8)
    axs[0,0].set_yticks([500, 750, 1000, 1250])

    axs[0,1].plot(years, P, color = 'black')
    axs[0,1].set_yticks([10000,20000,30000])

    axs[1,0].plot(years, y, color = 'black')
    axs[1,0].plot(years, A_, color = 'black', alpha = 0.8)

    axs[1,0].set_yticks([500, 750, 1000, 1250])

    axs[1,1].plot(years, P_, color = 'black')
    axs[1,1].set_yticks([2500, 5000, 7500, 10000])

    for i in range(2):
        for j in range(2):
            axs[i,j].spines['right'].set_visible(False)
            axs[i,j].spines['top'].set_visible(False)
            axs[i,j].set_xticks(years_xlabel)
    fig.suptitle("Filtering and smoothing output when observations are missing", y = 0.0, fontsize = 25)
    plt.savefig(figures_dir / 'Fig_2_5.png')


if __name__ == "__main__":
    main()
