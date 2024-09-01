import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import pandas as pd
import seaborn as sns

import sys
from pathlib import Path
current_script_path = Path(__file__).resolve()

functions_dir = current_script_path.parents[1] / 'Functions'
sys.path.insert(0, str(functions_dir))
import mouse as mouse

ssm_dir = current_script_path.parents[2] / 'SSM'
sys.path.insert(0, str(ssm_dir))
import ssm as ssm
from ssm.plots import gradient_cmap

aeon_mecha_dir = current_script_path.parents[2] / 'aeon_mecha' 
sys.path.insert(0, str(aeon_mecha_dir))
import aeon
import aeon.io.api as api
from aeon.io import reader, video
from aeon.schema.schemas import social02

nodes_name = ['nose', 'head', 'right_ear', 'left_ear', 'spine1', 'spine2','spine3', 'spine4']
color_names = [
    'black', 'blue', 'red', 'tan', 'green', 'brown', 
    'purple', 'orange', 'magenta', 'olive', 'pink', 
    'darkblue', 'lime', 'cyan', 'turquoise', 'gold', 
    'navy', 'maroon', 'teal', 'grey']

def main():

    Mouse = mouse.Mouse(aeon_exp = 'AEON3', type = 'Pre', mouse = 'BAA-1104047')
    Mouse.Run_Visits()

    NODES = [['head', 'spine3'],['spine1', 'spine3'],['left_ear', 'spine3'],['right_ear', 'spine3']]
    for nodes in NODES:
        Mouse.Add_Body_Info_to_mouse_pos(property = 'distance', nodes = nodes)

    root = Mouse.root
    
    Mouse.hmm.Get_States(n_state = 10, feature = 'Kinematics_and_Body')
    N = Mouse.hmm.n_state
    Pellets = Mouse.arena.pellets.index
    Visits = Mouse.arena.visits.dropna(subset=['speed'])
    Starts = Visits['start']
    Ends = Visits['end']
    Entry = Mouse.arena.entry 

    Mouse.mouse_pos['state'] = Mouse.hmm.states
    kl_matrix = np.log10(Mouse.hmm.kl_divergence+1)

    colors = sns.xkcd_palette(color_names[0:N])
    cmap = gradient_cmap(colors)
    
    folder_name = 'StartVisit'
    fig, axs = plt.subplots(1,4,figsize = (30, 8))

    times = [-1, -0.3, 0.1, 2]
    trigger = Mouse.arena.visits['start'].to_numpy()[-48]
    
    for i in range(len(times)):
        time = trigger + pd.Timedelta(str(times[i]) + "S")
        start, end = time, time + pd.Timedelta("1S")
        state = Mouse.mouse_pos.loc[start:end, 'state'][0]
        video_metadata = aeon.load(root, social02.CameraPatch3.Video, start=start, end=end)
        video_metadata.index = video_metadata.index.round("20L")
        frames = video.frames(video_metadata)
        first_frame = next(frames)
        rgb_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        axs[i].imshow(rgb_frame)
        square_size = 0.2
        rect_x = 1 - square_size
        rect_y = 1 - square_size
        rect = patches.Rectangle((1 - square_size, 1 - square_size), square_size, square_size, 
                        transform=axs[i].transAxes, color=color_names[state])
        axs[i].add_patch(rect)
        axs[i].text(rect_x + square_size/2, rect_y + square_size/2, str(state+1),
                transform=axs[i].transAxes, 
                ha='center', va='center', 
                color='white', fontweight='bold', fontsize=36)
        axs[i].set_title('Time = ' + str(times[i]) + ' s, State = ' + str(state+1), fontsize = 30)
        axs[i].axis('off')
    plt.tight_layout()
    plt.savefig('../Images/EventFrames/'+ folder_name +'.png')

    folder_name = 'PelletDelivery'
    fig, axs = plt.subplots(1,4,figsize = (30, 8))

    times = [0, 1, 1.5, 3]
    trigger = Mouse.arena.pellets.index[486]

    for i in range(len(times)):
        time = trigger + pd.Timedelta(str(times[i]) + "S")
        start, end = time, time + pd.Timedelta("1S")
        state = Mouse.mouse_pos.loc[start:end, 'state'][0]
        video_metadata = aeon.load(root, social02.CameraPatch3.Video, start=start, end=end)
        video_metadata.index = video_metadata.index.round("20L")
        frames = video.frames(video_metadata)
        first_frame = next(frames)
        rgb_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        axs[i].imshow(rgb_frame)
        square_size = 0.2
        rect = patches.Rectangle((1 - square_size, 1 - square_size), square_size, square_size, 
                        transform=axs[i].transAxes, color=color_names[state])
        axs[i].text(rect_x + square_size/2, rect_y + square_size/2, str(state+1),
                transform=axs[i].transAxes, 
                ha='center', va='center', 
                color='white', fontweight='bold', fontsize=36)
        axs[i].add_patch(rect)
        axs[i].set_title('Time = ' + str(times[i]) + ' s, State = ' + str(state+1), fontsize = 30)
        axs[i].axis('off')
    plt.tight_layout()
    plt.savefig('../Images/EventFrames/'+ folder_name +'.png')
    
    folder_name = 'EndVisit'
    fig, axs = plt.subplots(1,4,figsize = (30, 8))

    times = [-3, -1.5, -0.3, 0]
    trigger = Mouse.arena.visits['end'].to_numpy()[89]

    for i in range(len(times)):
        time = trigger + pd.Timedelta(str(times[i]) + "S")
        start, end = time, time + pd.Timedelta("1S")
        state = Mouse.mouse_pos.loc[start:end, 'state'][0]
        video_metadata = aeon.load(root, social02.CameraPatch3.Video, start=start, end=end)
        video_metadata.index = video_metadata.index.round("20L")
        frames = video.frames(video_metadata)
        first_frame = next(frames)
        rgb_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        axs[i].imshow(rgb_frame)
        square_size = 0.2
        rect = patches.Rectangle((1 - square_size, 1 - square_size), square_size, square_size, 
                        transform=axs[i].transAxes, color=color_names[state])
        axs[i].add_patch(rect)
        axs[i].text(rect_x + square_size/2, rect_y + square_size/2, str(state+1),
                transform=axs[i].transAxes, 
                ha='center', va='center', 
                color='white', fontweight='bold', fontsize=36)
        axs[i].set_title('Time = ' + str(times[i]) + ' s, State = ' + str(state+1), fontsize = 30)
        axs[i].axis('off')
    plt.tight_layout()
    plt.savefig('../Images/EventFrames/'+ folder_name +'.png')
        
    
if __name__ == "__main__":
    main()
        