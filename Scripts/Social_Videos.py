import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib.animation import FuncAnimation, FFMpegWriter

import sys
from pathlib import Path
current_script_path = Path(__file__).resolve()

functions_dir = current_script_path.parents[1] / 'Functions'
sys.path.insert(0, str(functions_dir))
import mouse as mouse

aeon_mecha_dir = current_script_path.parents[2] / 'aeon_mecha' 
sys.path.insert(0, str(aeon_mecha_dir))
import aeon
import aeon.io.api as api
from aeon.io import reader, video
from aeon.schema.schemas import social02
from aeon.analysis.utils import visits, distancetravelled

color_names = {
    0: (0, 0, 0),       # Black
    1: (0, 0, 255),     # Blue (BGR for Blue)
    2: (255, 0, 0),     # Red (BGR for Red)
    3: (140, 180, 210), # Tan (BGR for Tan)
    4: (0, 255, 0),     # Green (BGR for Green)
    5: (42, 42, 165),   # Brown (BGR for Brown)
    6: (128, 0, 128),   # Purple (BGR for Purple)
    7: (0, 165, 255),   # Orange (BGR for Orange)
    8: (255, 0, 255),   # Magenta (BGR for Magenta)
    9: (0, 128, 128),   # Olive (BGR for Olive)
    10: (203, 192, 255),# Pink (BGR for Pink)
    11: (139, 0, 0),    # Dark Blue (BGR for Dark Blue)
    12: (0, 255, 0),    # Lime (BGR for Lime)
    13: (255, 255, 0),  # Cyan (BGR for Cyan)
    14: (255, 0, 255),  # Magenta (BGR for Magenta)
    15: (0, 215, 255),  # Gold (BGR for Gold)
    16: (128, 0, 0),    # Navy (BGR for Navy)
    17: (0, 0, 128),    # Maroon (BGR for Maroon)
    18: (128, 128, 0),  # Teal (BGR for Teal)
    19: (128, 128, 128) # Grey (BGR for Grey)
}

def Export_Video_Nest(root, start, end, file_name):
    video_metadata = aeon.load(root, social02.CameraNest.Video, start=start, end=end)
    
    video_metadata.index = video_metadata.index.round("20L")  # round timestamps to nearest 20 ms
    frames = video.frames(video_metadata)  # get actual frames based on vid metadata
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid = cv2.VideoWriter("../Videos/" + file_name + ".mp4", fourcc=fourcc, fps=50, frameSize=(1440, 1080))  # will save to current dir
    for f in frames:  # write out frames to vid, frame-by-frame
        vid.write(f)
    vid.release()
    
def Export_Video_Nest_with_State(root, start, end, states, file_name):
    states = np.repeat(states[:30], 5)
    
    video_metadata = aeon.load(root, social02.CameraNest.Video, start=start, end=end)
    video_metadata.index = video_metadata.index.round("20L")  # round timestamps to nearest 20 ms
    video_metadata = video_metadata[~video_metadata.index.duplicated(keep='first')][:150]

    frames = video.frames(video_metadata)  # get actual frames based on vid metadata
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid = cv2.VideoWriter("../Videos/" + file_name + ".mp4", fourcc=fourcc, fps=50, frameSize=(1440, 1080))  # will save to current dir
    
    for i, f in enumerate(frames):
        # Get the corresponding color for the current state
        color = color_names[states[i]]
        
        # Draw the solid square on the top right
        top_left_corner = (1240, 40)  # Adjust coordinates as needed
        bottom_right_corner = (1400, 200)  # Adjust square size as needed
        cv2.rectangle(f, top_left_corner, bottom_right_corner, color, -1)  # -1 fills the rectangle
        
        # Write the modified frame to the video
        vid.write(f)
    vid.release() 
    
    
def Export_Video_Patch(root, patch, start, end, file_name):
    if patch == 'Patch1':
        video_metadata = aeon.load(root, social02.CameraPatch1.Video, start=start, end=end)
    if patch == 'Patch2':
        video_metadata = aeon.load(root, social02.CameraPatch2.Video, start=start, end=end)
    if patch == 'Patch3':
        video_metadata = aeon.load(root, social02.CameraPatch3.Video, start=start, end=end)
    video_metadata.index = video_metadata.index.round("10L")  # round timestamps to nearest 10 ms
    video_metadata = video_metadata[~video_metadata.index.duplicated(keep='first')][:150]
    frames = video.frames(video_metadata)  # get actual frames based on vid metadata
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid = cv2.VideoWriter("../Videos/" + file_name + ".mp4", fourcc=fourcc, fps=50, frameSize=(720, 540))  # will save to current dir
    for f in frames:  # write out frames to vid, frame-by-frame
        vid.write(f)
    vid.release()
    
def Export_Video_Patch_with_State(root, patch, start, end, states, file_name):
    if patch == 'Patch1':
        video_metadata = aeon.load(root, social02.CameraPatch1.Video, start=start, end=end)
    if patch == 'Patch2':
        video_metadata = aeon.load(root, social02.CameraPatch2.Video, start=start, end=end)
    if patch == 'Patch3':
        video_metadata = aeon.load(root, social02.CameraPatch3.Video, start=start, end=end)
    video_metadata.index = video_metadata.index.round("10L")  # round timestamps to nearest 10 ms
    video_metadata = video_metadata[~video_metadata.index.duplicated(keep='first')][:300]
    
    states = np.repeat(states[:30], 10) 
    
    frames = video.frames(video_metadata)  # get actual frames based on vid metadata
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid = cv2.VideoWriter("../Videos/" + file_name + ".mp4", fourcc=fourcc, fps=50, frameSize=(720, 540))  # will save to current dir
    ''' 
    for f in frames:  # write out frames to vid, frame-by-frame
        vid.write(f)
    '''
    for i, f in enumerate(frames):
        # Get the corresponding color for the current state
        color = color_names[states[i]]
        
        # Draw the solid square on the top right
        top_left_corner = (620, 20)  # Adjust coordinates as needed 
        bottom_right_corner = (700, 100)  # Adjust square size as needed
        cv2.rectangle(f, top_left_corner, bottom_right_corner, color, -1)  # -1 fills the rectangle
        
        # Write the modified frame to the video
        vid.write(f)
    vid.release()

def Get_Event_State(Mouse, trigger, right_seconds):
    right_period = pd.Timedelta(str(right_seconds+1) + 'S')
    
    next_valid_index = Mouse.mouse_pos.loc[trigger:trigger + right_period, 'state'].index
    next_valid_variable = Mouse.mouse_pos.loc[next_valid_index, ['state']].values.reshape(-1)
    if len(next_valid_variable) >= 10*right_seconds: next_valid_variable  = next_valid_variable[:10*right_seconds]
    
    return next_valid_variable
    
def main():
    
    Mouse = mouse.Mouse(aeon_exp = 'AEON3', type = 'Pre', mouse = 'BAA-1104045')
    states = np.load('../../SocialData/HMMStates/States_Pre_BAA-1104045.npy', allow_pickle=True)
    Mouse.Run_Visits()

    Mouse.mouse_pos['state'] = pd.Series(states, index = Mouse.mouse_pos.index)
    patch_loc = [Mouse.arena.patch_location['Patch' + str(i+1)] for i in range(3)]
    
    root = Mouse.root
    
    
    '''# nest state 2
    trigger = Mouse.mouse_pos.index[64130]
    start, end = trigger - pd.Timedelta("1S"), trigger + pd.Timedelta("2S")
    Export_Video_Nest(root, start, end, 'NestState2')
    print('NestState2')
    
    
    # nest state 5
    trigger = Mouse.mouse_pos.index[21427]
    start, end = trigger - pd.Timedelta("1S"), trigger + pd.Timedelta("2S")
    Export_Video_Nest(root, start, end, 'NestState5')
    print('NestState5')
    
    # nest state 6
    trigger = Mouse.mouse_pos.index[155497]
    start, end = trigger - pd.Timedelta("1S"), trigger + pd.Timedelta("2S")
    Export_Video_Nest(root, start, end, 'NestState6')
    print('NestState6')
    
    '''
    
    idx = [132418, 155487, 204605, 1000122, 1016950, 1164486, 1243718, 1538474, 1716560, 1766356, 1800827, 1991241, 1994052, 2034188, 2034323, 2174618]
    for i in range(len(idx)):
        trigger = Mouse.mouse_pos.index[idx[i]]
        states = Get_Event_State(Mouse, trigger, 3)

        start, end = trigger, trigger + pd.Timedelta("3S")

        Export_Video_Nest_with_State(root, start, end, states, 'video' + str(i+1))
    '''
    
    
    
    # arena state 2
    trigger = Mouse.mouse_pos.index[20660]
    start, end = trigger - pd.Timedelta("1S"), trigger + pd.Timedelta("2S")
    Export_Video_Patch(root, 'Patch3', start, end, 'ArenaState2')
    print('ArenaState2')
    
    # arena state 5
    trigger = Mouse.mouse_pos.index[25938]
    start, end = trigger - pd.Timedelta("1S"), trigger + pd.Timedelta("2S")
    Export_Video_Patch(root, 'Patch2', start, end, 'ArenaState5')
    print('ArenaState5')
    
    arena_state_5_idx = [25928, 106953, 929295, 979442, 1066113, 1098846, 1324523, 1623061, 1630265, 1631012, 1631760, 1677684]
    for i in range(len(arena_state_5_idx)):
        trigger = Mouse.mouse_pos.index[arena_state_5_idx[i]]
        states = Get_Event_State(Mouse, trigger)

        x = Mouse.mouse_pos.smoothed_position_x[arena_state_5_idx[i]]
        y = Mouse.mouse_pos.smoothed_position_y[arena_state_5_idx[i]]
        patch_distance = [(x-patch_loc[j][0])**2 + (y-patch_loc[j][1])**2 for j in range(3)]
        patch_idx = np.argsort(patch_distance)[0]
        patch = 'Patch' + str(patch_idx + 1)
    
        start, end = trigger, trigger + pd.Timedelta("3S")

        Export_Video_Patch_with_State(root, patch, start, end, states, 'video' + str(i+1))
    
    
    # PelletStates6_7
    #pellets_idx = [12, 23, 72, 94, 110, 115, 156, 166, 196, 241, 298, 334, 340, 405, 430, 490]
    pellets_idx = [12]
    for i in range(len(pellets_idx)):
        trigger = Mouse.arena.pellets.index[pellets_idx[i]]
        states = Get_Event_State(Mouse, trigger, 3)

        idx = Mouse.arena.visits['start'].searchsorted(trigger, side='right') - 1
        patch = Mouse.arena.visits['patch'][idx]
        start, end = trigger, trigger + pd.Timedelta("3S")

        Export_Video_Patch_with_State(root, patch, start, end, states, 'video' + str(i+1))

    '''
    
    
if __name__ == "__main__":
    main()
        