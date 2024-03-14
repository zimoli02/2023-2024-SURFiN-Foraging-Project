import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(parent_dir)
import Scripts.lds as lds

def main():
    a1, p1 = 0, 1e3
    n = 1469.1
    e = 15099
    
    y = lds.SimulateData(a1, p1, e, n, 100)
    
    # Parameter: e = \sigma_{\eta}^2
    L1, E1, grad1 = lds.MaximizeLikelihood(y, paras = dict({'a1':a1, 'p1':p1, 'e': 1, 'n':n}), para = 'e')
    L2, E2, grad2 = lds.MaximizeLikelihood(y, paras = dict({'a1':a1, 'p1':p1, 'e': 25000, 'n':n}), para = 'e')
    E3 = np.arange(1,25000)
    grad3 = [lds.dLikelihood(y, dict({'a1':a1, 'p1':p1, 'e': i, 'n':n}), para = 'e') for i in E3]
    
    fig, axs = plt.subplots(1,3, figsize = (28, 6))
    axs[0].plot(np.log(E1), L1, color = 'red')
    axs[0].plot(np.log(E2), L2, color = 'blue')
    axs[0].set_title("Log Likelihood", fontsize = 16)
    axs[0].set_ylabel("Log L", fontsize = 12)
    
    axs[1].plot(np.log(E1), grad1, color = 'red')
    axs[1].plot(np.log(E2), grad2, color = 'blue')
    axs[1].set_title("Gradient of Log Likelihood", fontsize = 16)
    axs[1].set_ylabel("d(Log L)", fontsize = 12)
    
    axs[2].plot(np.log(E3), grad3, color = 'black')
    axs[2].set_title("Gradient of Log Likelihood: dL/de", fontsize = 16)
    axs[2].set_ylabel("d(Log L)/de", fontsize = 12)
    
    for i in range(3): 
        axs[i].set_xlabel("ln(\sigma_e^2)", fontsize = 12)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
    
    plt.show()
    
    # Parameter: n = \sigma_{\epsilon}^2
    L1, N1, grad1 = lds.MaximizeLikelihood(y, paras = dict({'a1':a1, 'p1':p1, 'e': e, 'n':1}), para = 'n')
    L2, N2, grad2 = lds.MaximizeLikelihood(y, paras = dict({'a1':a1, 'p1':p1, 'e': e, 'n':10000}), para = 'n')
    N3 = np.arange(1,10000)
    grad3 = [lds.dLikelihood(y, dict({'a1':a1, 'p1':p1, 'e': e, 'n':i}), para = 'n') for i in N3]
    
    fig, axs = plt.subplots(1,3, figsize = (28, 6))
    axs[0].plot(np.log(N1), L1, color = 'red')
    axs[0].plot(np.log(N2), L2, color = 'blue')
    axs[0].set_title("d Log Likelihood/dn", fontsize = 16)
    axs[0].set_ylabel("Log L", fontsize = 12)
    
    axs[1].plot(np.log(N1), grad1, color = 'red')
    axs[1].plot(np.log(N2), grad2, color = 'blue')
    axs[1].set_title("Gradient of Log Likelihood: dL", fontsize = 16)
    axs[1].set_ylabel("d(Log L)", fontsize = 12)
    
    axs[2].plot(np.log(N3), grad3, color = 'black')
    axs[2].set_title("Gradient of Log Likelihood: dL/dn", fontsize = 16)
    axs[2].set_ylabel("d(Log L)/dn", fontsize = 12)
    
    for i in range(3): 
        axs[i].set_xlabel("ln(\sigma_n^2)", fontsize = 12)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
    
    plt.show()

if __name__ == "__main__":
    main()
