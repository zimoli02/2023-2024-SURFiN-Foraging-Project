import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

import sys
from pathlib import Path

aeon_mecha_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(aeon_mecha_dir))


color_names = ["blue","red","yellow", "green","brown","purple","orange","black"]


def PlotTransM(title):
    TransM = np.load('../Data/HMMStates/' + title + "TransM_Unit.npy", allow_pickle=True)
    
    N = len(TransM)
    fig, ax = plt.subplots()
    colors = color_names[0:N]

    # Draw rectangles for states
    # Assuming a fixed width of 0.8 for the 4:3 ratio, calculate the height
    width = 0.4
    height = 0.3   # This maintains a 3:4 ratio

    # To increase distance between rectangles, modify the y-coordinate calculation
    # Let's add an additional space of 0.1 between each rectangle
    spacing = 0.5  # Additional spacing between rectangles
    

    for i in range(N):
        y_position = (N-1-i) * (height + spacing)  # Adjusted y position to include spacing
        rect = patches.Rectangle((0.1, y_position), width, height, linewidth=1, edgecolor=None, facecolor=colors[i])
        ax.add_patch(rect)
        ax.text(0.3, y_position + height/2, f'State {i}', ha='center', va='center', color='black', fontsize = 16)

    # Identify and draw highest probability transitions
    for i in range(N):
        j1 = np.argmax(TransM[i])  # Highest prob from i to j
        j2 = np.argmax(TransM[:,i])  # Highest prob to i from j
        
        # Probability values
        prob_ij1 = TransM[i, j1]
        prob_j2i = TransM[j2, i]

        # Draw arrows and label them

        ax.annotate('', xy=(0.5, N-0.5-j1), xytext=(0.5, 5-0.5-i),
                        arrowprops=dict(arrowstyle="->", color='black'),
                        )
        ax.text(0.6, 5-0.25*(i+j1), f'{prob_ij1:.2f}', color='black')


        ax.annotate('', xy=(0.5, 5-0.5-i), xytext=(0.5, 5-0.5-j2),
                        arrowprops=dict(arrowstyle="->", color='black'),
                        )
        ax.text(0.6, N-0.25*(i+j2), f'{prob_j2i:.2f}', color='black')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, N)
    ax.axis('off')
    plt.savefig('../Images/HMM_TransitionM/' + title +'.png')
    plt.show()

def PlotTransMShort(title):
    PlotTransM(title)


def PlotTransMLong():
    PlotTransM()
    

def main():
    PlotTransMShort(title = 'ShortSession5')
    #PlotTransMLong()
    

if __name__ == "__main__":
    main()