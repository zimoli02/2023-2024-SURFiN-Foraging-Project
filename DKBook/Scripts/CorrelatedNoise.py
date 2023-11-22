import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lds

def main():
    a1, p1 = 1, 0
    e, w = 15000, 100
    n = 1500

    row, column = 2, 1
    fig, axs = plt.subplots(row, column,  figsize = (10*column-2, 7*row-2))

    for i in range(100):
        y, E = lds.SimulateData_CorrelateNoise(a1, p1, e, n, w, num=100)

        axs[0].plot(y, color = 'red', alpha = 0.1)
        axs[0].set_ylabel("Simulated Observations")

        axs[1].plot(E, color = 'blue', alpha = 0.1)
        axs[1].set_ylabel("Observation Noise")

    plt.savefig('CorrelatedNoise.png')



if __name__ == "__main__":
    main()
