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

import os
import subprocess

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
    1: (255, 0, 0),     # Blue (BGR for Blue)
    2: (0, 0, 255),     # Red (BGR for Red)
    3: (180, 210, 240), # Tan (BGR for Tan)
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

def combine_videos(input_dir, output_dir, row, column, output_filename, scale):
    # Get all mp4 files in the input directory
    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.mp4')])
    
    if not input_files:
        print("No MP4 files found in the input directory.")
        return

    # Prepare the ffmpeg command
    command = ['ffmpeg']
    
    # Add input files
    for file in input_files:
        command.extend(['-i', os.path.join(input_dir, file)])
    
    # Prepare filter complex
    filter_complex = []
    for i, _ in enumerate(input_files):
        filter_complex.append(f'[{i}:v] setpts=PTS-STARTPTS, scale={scale} [a{i}];')
    
    # Create hstack and vstack
    num_rows = row
    for i in range(num_rows):
        start = i * column
        end = min(start + column, len(input_files))
        inputs = ''.join(f'[a{j}]' for j in range(start, end))
        filter_complex.append(f'{inputs} hstack=inputs={end-start} [b{i}];')
    
    vstack_inputs = ''.join(f'[b{i}]' for i in range(num_rows))
    filter_complex.append(f'{vstack_inputs} vstack=inputs={num_rows} [outv]')
    
    # Add filter complex to command
    command.extend(['-filter_complex', ' '.join(filter_complex)])
    
    # Add output options
    output_path = os.path.join(output_dir, output_filename)
    command.extend(['-map', '[outv]', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', output_path])
    
    # Execute the command
    try:
        subprocess.run(command, check=True)
        print(f"Videos combined successfully. Output saved as {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while combining videos: {e}")

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
    vid = cv2.VideoWriter("../Videos/SubVideos/" + file_name + ".mp4", fourcc=fourcc, fps=50, frameSize=(1440, 1080))  # will save to current dir
    
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
    
    # Adjust length
    video_metadata = video_metadata[~video_metadata.index.duplicated(keep='first')][:300]
    states = np.repeat(states[:30], 10) 
    
    frames = video.frames(video_metadata)  # get actual frames based on vid metadata
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid = cv2.VideoWriter("../Videos/SubVideos/" + file_name + ".mp4", fourcc=fourcc, fps=50, frameSize=(720, 540))  # will save to current dir
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
    input_dir = Path("../Videos/SubVideos")
    output_dir = Path("../Videos")
    
    # nest state 1
    '''idx = [387221, 535534, 619131, 639646, 818841, 1233484, 1405769, 1518902, 1806994, 1835481, 2081265, 2144370, 2228862, 2272409, 2304439, 2509486] # Mouse Pre 045
    for i in range(len(idx)):
        trigger = Mouse.mouse_pos.index[idx[i]]
        states = Get_Event_State(Mouse, trigger, 3)
        start, end = trigger, trigger + pd.Timedelta("3S")
        Export_Video_Nest_with_State(root, start, end, states, 'video' + str(i+1))
        print('Export '+ 'video' + str(i+1))
    combine_videos(input_dir, output_dir, 4, 4, "NestState1_All.mp4", scale = '1440x1080')
    print('NestState1')
    
    idx = [1233484] # Mouse Pre 045
    for i in range(len(idx)):
        trigger = Mouse.mouse_pos.index[idx[i]]
        states = Get_Event_State(Mouse, trigger, 3)
        start, end = trigger, trigger + pd.Timedelta("3S")
        Export_Video_Nest_with_State(root, start, end, states, 'NestState1')
        print('Export '+ 'video' + str(i+1))
    '''
    # nest state 2
    '''
    idx = [580639, 591269,609768,613907,635678,665539,1180916,1247465,1315053,1347723,1611586,1970248,2039672,2041919,2134672,2329307] # Mouse Pre 045
    for i in range(len(idx)):
        trigger = Mouse.mouse_pos.index[idx[i]]
        states = Get_Event_State(Mouse, trigger, 3)
        start, end = trigger, trigger + pd.Timedelta("3S")
        Export_Video_Nest_with_State(root, start, end, states, 'video' + str(i+1))
        print('Export '+ 'video' + str(i+1))
    
    combine_videos(input_dir, output_dir, 4, 4, "NestState2_All.mp4", scale = '1440x1080')
    print('NestState2')
    
    
    idx = [2041919] # Mouse Pre 045
    for i in range(len(idx)):
        trigger = Mouse.mouse_pos.index[idx[i]]
        states = Get_Event_State(Mouse, trigger, 3)
        start, end = trigger, trigger + pd.Timedelta("3S")
        Export_Video_Nest_with_State(root, start, end, states, 'NestState2')
        print('Export '+ 'video' + str(i+1))
    '''    
    # nest state 6
    '''
    idx = [21418, 354275, 462567, 563975, 701801, 758035, 882169, 1152403, 1317760, 1465204, 1569692, 1719294, 1850512, 2033279, 2333054, 2508854] # Mouse Pre 045
    for i in range(len(idx)):
        trigger = Mouse.mouse_pos.index[idx[i]]
        states = Get_Event_State(Mouse, trigger, 3)
        start, end = trigger, trigger + pd.Timedelta("3S")
        Export_Video_Nest_with_State(root, start, end, states, 'video' + str(i+1))
        print('Export '+ 'video' + str(i+1))
    
    combine_videos(input_dir, output_dir, 4, 4, "NestState6_All.mp4", scale = '1440x1080')
    print('NestState6')
    
    
    idx = [563975, 701801, 1465204, 1569692, 1719294, 2033279] # Mouse Pre 045
    for i in range(len(idx)):
        trigger = Mouse.mouse_pos.index[idx[i]]
        states = Get_Event_State(Mouse, trigger, 3)
        start, end = trigger, trigger + pd.Timedelta("3S")
        Export_Video_Nest_with_State(root, start, end, states, 'video' + str(i+1))
        print('Export '+ 'video' + str(i+1))
    
    combine_videos(input_dir, output_dir, 2, 3, "NestState6_Some.mp4", scale = '1440x1080')
    print('NestState6')
    '''

    # arena state 2
    '''
    trigger = Mouse.mouse_pos.index[20660] # Mouse Pre 045
    start, end = trigger - pd.Timedelta("1S"), trigger + pd.Timedelta("2S")
    Export_Video_Patch(root, 'Patch3', start, end, 'ArenaState2')
    print('ArenaState2')
    '''
    
    
    # arena state - grooming
    
    idx = [25327,108123,928695,1065510,1528816,1620069,1627275,1628765,1674693,1678561,1698418,1849673,2542977] # Mouse Pre 045
    #idx = [25928,106952,1631012,1630265, 1631760, 108729,929296,979442] # Mouse Pre 045
    '''idx = [849618, 1582605, 1649143] # Mouse Post 047'''
    for i in range(9):
        trigger = Mouse.mouse_pos.index[idx[i]]
        states = Get_Event_State(Mouse, trigger, 3)

        x = Mouse.mouse_pos.smoothed_position_x[idx[i]]
        y = Mouse.mouse_pos.smoothed_position_y[idx[i]]
        patch_distance = [(x-patch_loc[j][0])**2 + (y-patch_loc[j][1])**2 for j in range(3)]
        patch_idx = np.argsort(patch_distance)[0]
        patch = 'Patch' + str(patch_idx + 1)
    
        start, end = trigger, trigger + pd.Timedelta("3S")

        Export_Video_Patch_with_State(root, patch, start, end, states, 'video' + str(i+1))
        print('Export '+ 'video' + str(i+1))
        
    combine_videos(input_dir, output_dir, 3, 3, "ArenaStateGrooming_Pre_045.mp4", scale = '720x540')
    print('ArenaStateGrooming')
    
    '''
    # Start Visit
    idx = [19, 28, 101, 121, 131, 46, 47, 139, 104] # Mouse 045
    for i in range(len(idx)):
        trigger = Mouse.arena.visits['start'][idx[i]] - pd.Timedelta("1.5S")
        states = Get_Event_State(Mouse, trigger, 3)

        x = Mouse.mouse_pos.smoothed_position_x[idx[i]]
        y = Mouse.mouse_pos.smoothed_position_y[idx[i]]
        patch_distance = [(x-patch_loc[j][0])**2 + (y-patch_loc[j][1])**2 for j in range(3)]
        patch_idx = np.argsort(patch_distance)[0]
        patch = 'Patch' + str(patch_idx + 1)
    
        start, end = trigger, trigger + pd.Timedelta("3S")

        Export_Video_Patch_with_State(root, patch, start, end, states, 'video' + str(i+1))
        print('Export '+ 'video' + str(i+1))
        
    combine_videos(input_dir, output_dir, 3, 3, "StartVisit_All_045.mp4", scale = '720x540')
    print('StartVisit_All_045')
    '''
    
    # PelletStates
    '''
    pellets_idx = [12, 23, 72, 94, 110, 115, 156, 166, 196, 241, 298, 334, 340, 405, 430, 490] # Mouse Pre 045
    pellets_idx = [132, 162, 245, 255, 302, 326, 381, 390, 425, 440, 456, 482, 494, 497, 514, 532, 536, 633, 635, 643] # Mouse Post 047
    for i in range(len(pellets_idx[:16])):
        trigger = Mouse.arena.pellets.index[pellets_idx[i]]
        states = Get_Event_State(Mouse, trigger, 3)

        idx = Mouse.arena.visits['start'].searchsorted(trigger, side='right') - 1
        patch = Mouse.arena.visits['patch'][idx]
        start, end = trigger, trigger + pd.Timedelta("3S")

        Export_Video_Patch_with_State(root, patch, start, end, states, 'video' + str(i+1))
        print('Export '+ 'video' + str(i+1))
    combine_videos(input_dir, output_dir, 4, 4, "PelletState_All_047.mp4", scale = '720x540')
    print('PelletState_All')
    '''
    
    # PelletState6_7
    '''
    pellets_idx = [11, 12, 107, 122, 150, 189, 202, 283, 293, 307, 334, 337, 431, 539, 550, 556] # Mouse Pre 045
    pellets_idx = [2, 8, 12, 61, 114, 161, 165, 244, 259, 274, 285, 289, 328, 390, 435, 440, 445, 448, 465, 489, 514, 600, 647, 678] # Mouse Post 047
    for i in range(len(pellets_idx[0:16])):
        trigger = Mouse.arena.pellets.index[pellets_idx[i]]
        states = Get_Event_State(Mouse, trigger, 3)

        idx = Mouse.arena.visits['start'].searchsorted(trigger, side='right') - 1
        patch = Mouse.arena.visits['patch'][idx]
        start, end = trigger, trigger + pd.Timedelta("3S")

        Export_Video_Patch_with_State(root, patch, start, end, states, 'video' + str(i+1))
        print('Export '+ 'video' + str(i+1))
        
    combine_videos(input_dir, output_dir, 4, 4, "PelletState_Transit_All_047.mp4", scale = '720x540')
    print('PelletState6_7')'''
    
    
if __name__ == "__main__":
    main()
        