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

    a1, p1 = 1,0.1
    e = 1
    n = 0.1

    y = lds.SimulateData_Poisson(a1, p1, n, num=len(years))

    # Filter data
    a, p, F, A, P = lds.ApplyFilter(a1, p1, e, n, y)
    v = [y[i] - a[i] for i in range(len(y))]
    E, (S,p_S), (K,p_K), (NN, p_NN), (H, p_H), c, (Q,p_Q) = lds.DiagnosticCheck(a1, p1, e, n, y)


    years_xlabel = [1880, 1900, 1920, 1940, 1960]
    
    fig, axs = plt.subplots(2,2, figsize = (18,14))
    zeroline = np.zeros(len(y))
    axs[0,0].plot(years[1:], E, color = 'black')
    axs[0,0].plot(years, zeroline, color = 'grey')
    #axs[0,0].set_yticks([-2,0,2])
    axs[0,0].set_xticks(years_xlabel)
    axs[0,0].set_title('Standardized Residual')
    
    axs[0,1].hist(E, color = 'white', bins = 13, edgecolor = 'black', density = True)
    #axs[0,1].set_yticks([0.1,0.2,0.3,0.4,0.5])
    #axs[0,1].set_xticks([-3,-2,-1,0,1,2,3])
    axs[0,1].set_title('Histogram Plus Estimated Density')
    
    (QQ_x, QQ_y) = lds.TheoreticalQuantile(E)
    axs[1,0].plot(QQ_x, QQ_y, color = 'darkgrey')
    axs[1,0].plot(QQ_x, QQ_x, color = 'black', label = 'normal')
    axs[1,0].plot(QQ_x, np.zeros(len(E)), color = 'grey', linestyle = ':')
    #axs[1,0].set_yticks([-2,0,2])
    #axs[1,0].set_xticks([-2,-1,0,1,2])
    axs[1,0].set_title('Ordered Residuals')
    axs[1,0].legend(loc = 'upper left')
    
    axs[1,1].bar(np.arange(1,11), c, color = 'grey', edgecolor = 'black')
    #axs[1,1].set_yticks([-0.5, 0.0, 0.5, 1.0])
    #axs[1,1].set_xticks([0,5,10])
    axs[1,1].set_title('Correlogram')
    
    for i in range(2):
        for j in range(2):
            axs[i,j].spines['right'].set_visible(False)
            axs[i,j].spines['top'].set_visible(False)
    
    fig.suptitle("Nile data and output of Kalman filter", y = 0.0, fontsize = 25)
    plt.savefig(figures_dir / 'Fig_2_7_PoissonData.png')
    
    print('S = ', S, ' p value_S = ', round(p_S,4))
    print('K = ', K, ' p value_K = ', round(p_K,4))
    print('N = ', NN, ' p value_N = ', round(p_NN,4))
    print('H(33) = ', H[32], ' p value_H(33) = ', round(p_H[32],4))
    print('Q(9) = ', Q[8], ' p value_Q(9) = ', round(p_Q[8],4))

    
if __name__ == "__main__":
    main()
