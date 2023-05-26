# -*- coding: utf-8 -*-
"""
Created on Wed May 24 13:25:18 2023

@author: zafar
"""
from helper_parse import read_video, read_gaze
from helper_track import track_hands
from helper_export import export_video

VID_RES = [ 960, 540 ]
TRIAL_FRAMES = [ 1250, 1700 ] # start and end frames of trial
PATH_VIDEO = '../data/test-ewa/session.mp4'
PATH_GAZE = '../data/test-ewa/gaze_data.csv'
PATH_OUT_VIDEO = '../export/videos/'

# read video + data
video, fps, length = read_video(PATH_VIDEO, VID_RES)
raw_gaze, gaze_xy = read_gaze(PATH_GAZE, VID_RES)

# clip video + data to trial
trial_video = video[ TRIAL_FRAMES[0]:TRIAL_FRAMES[1] ]
trial_gaze_xy = gaze_xy[ TRIAL_FRAMES[0]:TRIAL_FRAMES[1], : ]

# track hands + gaze then export
_, _, _, out_frames = track_hands(trial_video, trial_gaze_xy, True)
export_video(PATH_OUT_VIDEO, "tol_pilot.mp4", out_frames, 15, VID_RES)
