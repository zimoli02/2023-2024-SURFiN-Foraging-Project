import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ApplyFilter(a1, p1, e, n, y, timepoints):
    a, p, F = np.zeros(timepoints+1, dtype = np.double), np.zeros(timepoints+1, dtype = np.double), np.zeros(timepoints, dtype = np.double)
    A, P = np.zeros(timepoints, dtype = np.double), np.zeros(timepoints, dtype = np.double)

    a[0], p[0] = a1, p1

    for i in range(0, timepoints):
        F[i] = p[i] + e
        K = p[i]/F[i]
        #predicting
        if np.isnan(y[i]): 
            K = 0
            A[i] = a[i]
            P[i] = p[i]
        else:
            A[i] = a[i] + K*(y[i] - a[i])
            P[i] = K*e
        #predicting
        a[i+1] = A[i]
        p[i+1] = P[i] + n
    
    return a[0:-1], p[0:-1],F,A,P

def ApplySmoothing(a, p, F, y, v, e, timepoints):
    r, N = np.zeros(timepoints, dtype = np.double), np.zeros(timepoints, dtype = np.double)
    A_, P_ = np.zeros(timepoints, dtype = np.double), np.zeros(timepoints, dtype = np.double)

    for i in range(timepoints-1, 0, -1):
        if np.isnan(y[i]): r[i-1], N[i-1] = r[i], N[i]
        else:
            r[i-1] = (v[i]/F[i]) + (e/F[i])*r[i]
            N[i-1] = (1/F[i]) + ((e/F[i])**2)*N[i]
    r_ = (v[0]/F[0]) + (e/F[0])*r[0]
    N_ = (1/F[0]) + ((e/F[0])**2)*N[0]
    
    A_[0] = a[0] + p[0]*r_
    P_[0] = p[0] - (p[0]**2)*N_
    for i in range(1,timepoints):
        A_[i] = a[i] + p[i]*r[i-1]
        P_[i] = p[i] - (p[i]**2)*N[i-1]
    
    return A_, P_, r, N

def main():
    # Import Data
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
    a,p,F, A, P = ApplyFilter(a1, p1, e, n, y, timepoints)


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
    plt.show()


if __name__ == "__main__":
    main()