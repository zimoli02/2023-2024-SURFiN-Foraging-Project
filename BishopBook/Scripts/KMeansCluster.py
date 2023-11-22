import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

import EM

def main():
  current_dir = Path(__file__).resolve().parent
  figures_dir = current_dir.parent / 'Figures'

  current_dir = os.path.dirname(os.path.abspath(__file__))
  OldFaithful = pd.read_csv(os.path.join(current_dir,'old_faithful.csv'))
  data = np.array([[i,j] for i, j in zip (OldFaithful['eruptions'].to_numpy(), OldFaithful['waiting'].to_numpy())])

  kmeans = EM.Cluster(data, colors = ['red', 'blue'], k=2)
  kmeans.Run()

  fig, axs = plt.subplots(2,1,figsize = (7, 16))

  axs[0] = kmeans.Plot_Data(axs = axs[0])
  axs[1].plot(np.arange(1,len(kmeans.J_)+1), kmeans.J_, color = 'black')
  axs[1].set_ylabel("J")
  axs[1].set_xlabel('Iteration')

  plt.savefig(figures_dir/'KMeans.png')

if __name__ == '__main__':
  main()
