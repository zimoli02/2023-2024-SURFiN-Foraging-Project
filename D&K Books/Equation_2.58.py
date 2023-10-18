import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lds

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(current_dir,'Niles.csv'))
    years = data['years'].to_numpy()
    y = data['y'].to_numpy()
    
    # Set Parameters
    a1, p1 = 0, 1e7
    n = 1469.1

    e = 500 #Forward
    L1, E1, grad1 = lds.MaximizeLikelihood(y, a1, p1, e, n)

    e = 25000 #Backward
    L2, E2, grad2 = lds.MaximizeLikelihood(y, a1, p1, e, n)
    
    fig, axs = plt.subplots(1,2, figsize = (18, 6))
    axs[0].plot(E1, L1, color = 'red')
    axs[0].plot(E2, L2, color = 'blue')
    axs[0].set_title("Log Likelihood", fontsize = 16)
    axs[0].set_xlabel("\sigma_e^2", fontsize = 14)
    axs[0].set_ylabel("Log L", fontsize = 14)

    axs[1].plot(E1, grad1, color = 'red')
    axs[1].plot(E2, grad2, color = 'blue')
    axs[1].set_title("Gradient of Log Likelihood", fontsize = 16)
    axs[1].set_xlabel("\sigma_e^2", fontsize = 14)
    axs[1].set_ylabel("d(Log L)/de", fontsize = 14)

    for i in range(2): 
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)

    plt.show()

    print("\sigma_e^2 at Max Log Likelihood: ", E1[-1])
    print("\sigma_e^2 at Max Log Likelihood: ", E2[-1])
    
if __name__ == "__main__":
    main()
