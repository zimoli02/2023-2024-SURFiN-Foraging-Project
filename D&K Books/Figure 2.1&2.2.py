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
        #filtering
        A[i] = a[i] + K*(y[i] - a[i])
        P[i] = K*e
        #predicting
        a[i+1] = A[i]
        p[i+1] = P[i] + n
    
    return a[0:-1], p[0:-1],F

def ApplySmoothing(a, p, F, v, e, timepoints):
    r, N = np.zeros(timepoints, dtype = np.double), np.zeros(timepoints, dtype = np.double)
    A_, P_ = np.zeros(timepoints, dtype = np.double), np.zeros(timepoints, dtype = np.double)

    for i in range(timepoints-1, 0, -1):
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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(current_dir,'Niles.csv'))
    years = data['years'].to_numpy()
    y = data['y'].to_numpy()

    a1, p1 = 0, 1e7
    e = 15099
    n = 1469.1
    timepoints = len(years)

    # Filter data
    a,p,F = ApplyFilter(a1, p1, e, n, y, timepoints)
    v = [y[i] - a[i] for i in range(timepoints)]

    # Plot filtered data
    # Note: here data is displayed from t = 2 to t = n
    years_xlabel = [1880, 1900, 1920, 1940, 1960]
    upper_bound = [a[i] + 1.65*(p[i]**0.5) for i in range(timepoints)]
    lower_bound = [a[i] - 1.65*(p[i]**0.5) for i in range(timepoints)]
    
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
    
    plt.show()

    # Smooth data
    A_, P_, r, N = ApplySmoothing(a, p, F, v, e, timepoints)
    t, end = 0,100
    years_xlabel = [1880, 1900, 1920, 1940, 1960]
    upper_bound = [A_[i] + 1.65*(P_[i]**0.5) for i in range(timepoints)]
    lower_bound = [A_[i] - 1.65*(P_[i]**0.5) for i in range(timepoints)]
    
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
    
    plt.show()

if __name__ == "__main__":
    main()
