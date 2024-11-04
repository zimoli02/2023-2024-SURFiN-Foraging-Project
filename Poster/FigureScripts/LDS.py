import cv2
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
from pathlib import Path

import sys
from pathlib import Path

aeon_mecha_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(aeon_mecha_dir))


import SURF_backup.UsedCode.kinematics as kinematics

start, end = 10*60*50+2220, 10*60*50+2240
dt = 0.02
scale = 2e-3

def main():
    # Display short session 0 as an example
    try:
        mouse_pos = pd.read_parquet('../FigureData/ShortSession0_mousepos.parquet', engine='pyarrow')
    except FileNotFoundError:
        mouse_pos = pd.read_parquet('../../Data/RawMouseKinematics/' + 'ShortSession0' + 'mousepos.parquet', engine='pyarrow')
        kinematics.AddKinematics('ShortSession0', mouse_pos) 
        mouse_pos.to_parquet('../FigureData/ShortSession0_mousepos.parquet', engine='pyarrow')
    
    x = mouse_pos.x[start:end] * scale
    x_vel = np.array([(x[i+1]-x[i])/dt for i in range(len(x)-1)])
    x_vel = np.concatenate((np.array([0]), x_vel))
    x_acce = np.array([(x_vel[i+1]-x_vel[i])/dt for i in range(len(x_vel)-1)])
    x_acce = np.concatenate((np.array([0]), x_acce))
        
    smooth_x = mouse_pos.smoothed_position_x[start:end] * scale
    smooth_x_var = mouse_pos.smoothed_position_x_var[start:end] * scale ** 2

    smooth_x_vel = mouse_pos.smoothed_velocity_x[start:end] * scale
    smooth_x_vel_var = mouse_pos.smoothed_velocity_x_var[start:end] * scale ** 2
    
    smooth_x_acce = mouse_pos.smoothed_acceleration_x[start:end] * scale
    smooth_x_acce_var = mouse_pos.smoothed_acceleration_x_var[start:end] * scale ** 2

    time = np.arange(0, len(x), 1)
    time = time * dt
        
    fig, axs = plt.subplots(1,3, figsize = (16,8))
    axs[0].plot(time, x, color = 'black', linewidth = 1, label = 'Raw')
    axs[0].scatter(time, x, color = 'black', s = 6)
    axs[0].plot(time, smooth_x, color = 'red', linewidth = 1, label = 'Smoothed')
    axs[0].scatter(time, smooth_x, color = 'red', s = 6)
    axs[0].fill_between(time, smooth_x - 1.65*(smooth_x_var**0.5), smooth_x + 1.65*(smooth_x_var**0.5), color = 'pink', alpha = 0.7, label = '95% C.I.')
    axs[0].legend(loc = 'upper right')

    axs[1].plot(time, x_vel, color = 'black', linewidth = 1, label = 'Raw')
    axs[1].scatter(time, x_vel, color = 'black', s = 6)
    axs[1].plot(time, smooth_x_vel, color = 'blue', linewidth = 1, label = 'Smoothed')
    axs[1].scatter(time, smooth_x_vel, color = 'blue', s=6)
    axs[1].fill_between(time, smooth_x_vel - 1.65*(smooth_x_vel_var**0.5), smooth_x_vel + 1.65*(smooth_x_vel_var**0.5), color = 'lightblue', alpha = 0.7, label = '95% C.I.')
    axs[1].legend(loc = 'lower right')
    
    axs[2].plot(time, x_acce, color = 'black', linewidth = 1, label = 'Raw')
    axs[2].scatter(time, x_acce, color = 'black', s = 6)
    axs[2].plot(time, smooth_x_acce, color = 'green', linewidth = 1, label = 'Smoothed')
    axs[2].scatter(time, smooth_x_acce, color = 'green', s=6)
    axs[2].fill_between(time, smooth_x_acce - 1.65*(smooth_x_acce_var**0.5), smooth_x_acce + 1.65*(smooth_x_acce_var**0.5), color = 'lightgreen', alpha = 0.7, label = '95% C.I.')
    axs[2].legend(loc = 'upper right')

    

    axs[0].set_ylabel('Position (m)', fontsize = 16)
    axs[1].set_ylabel('Speed (m/s)', fontsize = 16)
    axs[2].set_ylabel('Acceleration (m/s$^2$)', fontsize = 16)

    for i in range(3):
        axs[i].set_xticks(np.arange(0,time[-1]+0.01, 0.05))
        axs[i].set_xlabel('Time (s)', fontsize = 16)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        
        
    plt.tight_layout()
    plt.savefig('../FigureResults/LDS.png')
    plt.show()

if __name__ == "__main__":
        main()