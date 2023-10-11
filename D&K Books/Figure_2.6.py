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

        # Create Forcasting Points
    years_ = years
    y_ = np.array(y, dtype = float)
    for i in range(30): 
        y_ = np.append(y_, np.nan)
        years_ = np.append(years_, years_[-1] + 1)
    timepoints = len(y_)

    # Apply Filtering
    a, p, F, A, P = ApplyFilter(a1, p1, e, n, y_, timepoints)
    a_ = A
    p_ = [P[i] + e for i in range(len(P))]


    # Plotting
    years_xlabel = [1900, 1950, 2000]

    fig, axs = plt.subplots(2,2, figsize = (20,14))
    
    upper_bound = [A[i] + 0.67*(P[i]**0.5) for i in range(timepoints)]
    lower_bound = [A[i] - 0.67*(P[i]**0.5) for i in range(timepoints)]

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
    plt.show()


if __name__ == "__main__":
    main()