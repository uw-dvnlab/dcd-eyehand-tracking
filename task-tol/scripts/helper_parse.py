# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:41:40 2023

@author: zafar
"""
import cv2
import pandas as pd
import numpy as np

def read_video(path_video, vid_res):
    """
    Parameters
    ----------
    path_video : string
    vid_res : list (size 2)

    Returns
    -------
    video : list of video frames
    fps : int
    length : int
    """
    vidcap = cv2.VideoCapture(path_video)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    video = []

    for i in range(length):
        print(i)
        success, image = vidcap.read()
        if success:
            image = cv2.resize(image,
                               tuple(vid_res),
                               interpolation = cv2.INTER_AREA)
            video.append(image)

    return video, fps, length

def read_gaze(path_gaze, vid_res):
    """
    Parameters
    ----------
    path_gaze : string
    vid_res : list (size 2)

    Returns
    -------
    raw_gaze : DataFrame
    gaze_xy : array of gaze coordinates
    """
    # Read gaze file
    raw_gaze = pd.read_csv(path_gaze)
    # Get first gaze value for each frame index
    raw_gaze = raw_gaze.drop_duplicates('Frame_Index')
    # Structure gaze x/y/ position into array
    gaze_xy = np.array([raw_gaze.Image_X.values * vid_res[0],
                        raw_gaze.Image_Y.values * vid_res[1]]).astype(int).transpose()

    return raw_gaze, gaze_xy
