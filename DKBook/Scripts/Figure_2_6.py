import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(parent_dir)
import Scripts.lds as lds


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

        # Create Forcasting Points
    years_ = years
    y_ = np.array(y, dtype = float)
    for i in range(30): 
        y_ = np.append(y_, np.nan)
        years_ = np.append(years_, years_[-1] + 1)


    # Apply Filtering
    a, p, F, A, P = lds.ApplyFilter(a1, p1, e, n, y_)
    a_ = A
    p_ = [P[i] + e for i in range(len(P))]


    # Plotting
    years_xlabel = [1900, 1950, 2000]

    fig, axs = plt.subplots(2,2, figsize = (20,14))
    
    upper_bound = [A[i] + 0.67*(P[i]**0.5) for i in range(len(y_))]
    lower_bound = [A[i] - 0.67*(P[i]**0.5) for i in range(len(y_))]

    axs[0,0].scatter(years, y, color = 'black')
    axs[0,0].plot(years_, A, color = 'black', alpha = 0.8)
    axs[0,0].plot(years_[-30:-1], upper_bound[-30:-1], color = 'grey', alpha = 0.8)
    axs[0,0].plot(years_[-30:-1], lower_bound[-30:-1], color = 'grey', alpha = 0.8)
    axs[0,0].set_yticks([500, 750, 1000, 1250])

    axs[0,1].plot(years_, P, color = 'black')
    axs[0,1].set_yticks([10000,20000,30000, 40000, 50000])

    axs[1,0].plot(years_, a_, color = 'black', alpha = 0.8)
    axs[1,0].set_yticks([800, 900, 1000, 1100, 1200])

    axs[1,1].plot(years_, p_, color = 'black')
    axs[1,1].set_yticks([30000, 40000, 50000, 60000])

    for i in range(2):
        for j in range(2):
            axs[i,j].spines['right'].set_visible(False)
            axs[i,j].spines['top'].set_visible(False)
            axs[i,j].set_xticks(years_xlabel)
    fig.suptitle("Nile data and output of forecasting", y = 0.0, fontsize = 25)
    plt.savefig(figures_dir / 'Fig_2_6.png')


if __name__ == "__main__":
    main()
