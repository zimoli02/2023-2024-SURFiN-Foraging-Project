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

    plt.savefig(figures_dir / 'CorrelatedNoise.png')



if __name__ == "__main__":
    main()
