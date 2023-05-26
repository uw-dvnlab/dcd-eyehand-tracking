# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:32:51 2023

@author: zafar
"""
import cv2
import matplotlib.pyplot as plt
import pandas as pd

def export_video(frames_out, trial_no, path_export_video, fps, vid_res):
    print('Export Started')
    PATH_OUT = f'{path_export_video}/video_{trial_no}.mp4'
    video = cv2.VideoWriter(PATH_OUT, cv2.VideoWriter_fourcc(*'MP42'), fps, vid_res)
    for i in range(len(frames_out)):
        print(i)
        video.write(frames_out[i])

    video.release()
    print('Export Complete')
    
def export_figure(time_ot, time_ah, wrist_ot, wrist_ah, path_export_fig, trial_no):
    PATH_OUT = f'{path_export_fig}/figure_{trial_no}.png'

    plt.figure(figsize=(8, 4))
    plt.title(f'Finger X-Position: Trial {trial_no}')
    plt.plot(time_ot, wrist_ot, label='optotrak')
    plt.plot(time_ah, wrist_ah, label='video')
    plt.ylabel('Position (mm)')
    plt.xlabel('Time (sec)')
    plt.legend()
    plt.savefig(PATH_OUT)
    
def export_report(time, data, PATH_EXPORT_REPORT, filename):
    df = pd.DataFrame(data=[time, data]).T
    df.columns = ['time', 'x']

    df.to_csv(f'{PATH_EXPORT_REPORT}/{filename}')
    