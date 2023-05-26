# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 19:53:37 2023

@author: zafar
"""
import pandas as pd
import os
import cv2
import numpy as np

def read_video(path_video, vid_res):
    vidcap = cv2.VideoCapture(path_video)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    video = []

    for i in range(length):
        success,image = vidcap.read()
        if success:
            image = cv2.resize(image, tuple(vid_res), interpolation = cv2.INTER_AREA)
            video.append(image)
            
    return video, fps, length

def read_frames(trial_no, path_frames):
    raw_frames = pd.read_csv(path_frames)
    vid_start = raw_frames.frame_start.values[trial_no-1]
    vid_end = raw_frames.frame_end.values[trial_no-1]
    
    return raw_frames, vid_start, vid_end

def read_optotrak(trial_no, PATH_OT):
    # Get Optotrak Data
    FILE_OT = f"Pilot_ad_hock_fitts_2023_01_26_142916_{trial_no:03}_3d.csv"
    raw_ot = pd.read_csv(os.path.join(PATH_OT, FILE_OT), skiprows=4)
    cols_ot = raw_ot.columns[1:]
    raw_ot.drop(['wrist z'], axis=1, inplace=True)
    raw_ot.columns = cols_ot
    
    return raw_ot

def split_video(frames, vid_start, vid_end):
    return frames[vid_start:vid_end]

def read_gaze(vid_start, vid_end, vid_res, path_gaze):
    raw_gaze = pd.read_csv(path_gaze)
    
    raw_gaze = raw_gaze.drop_duplicates('Frame_Index')
    trial_gaze = raw_gaze.loc[(raw_gaze.Frame_Index >= vid_start) & (raw_gaze.Frame_Index < vid_end)]
    
    gaze_xy = np.array([trial_gaze.Image_X.values * vid_res[0], trial_gaze.Image_Y.values * vid_res[1]]).astype(int).transpose()
    
    return raw_gaze, gaze_xy