import numpy as np
import pandas as pd
import EM

def main():
  data = np.random.uniform(-1, 1, (100, 2))

  kmeans = Cluster(data, colors = ['red', 'blue', 'green', 'pink', 'yellow'], k=5)
  kmeans.Run()


if __name__ == '__main__':
  main()
