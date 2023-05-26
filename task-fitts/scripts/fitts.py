# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 17:47:40 2023

@author: zafar
"""
#%%
from helper_visualization import draw_out_frames
from helper_readdata import read_frames, read_optotrak, read_video, read_gaze, split_video
from helper_track import (track_hands, track_screen, get_dot_positions, adjust_gaze, get_hit_states, get_world_position_data)
from helper_export import export_video, export_figure, export_report

VID_RES = [960, 540]
PATH_VIDEO = '../data/adhawk/session.mp4'
PATH_FRAMES = '../data/adhawk/trial_frames.csv'
PATH_GAZE = '../data/adhawk/gaze_data.csv'
PATH_OT = '../data/optotrak'
PATH_EXPORT_VIDEO = '../export/videos'
PATH_EXPORT_FIG = '../export/figures'
PATH_EXPORT_REPORT = '../export/reports'

PX_SCREEN = [1920,1080] # px
CM_SCREEN = [52, 29.2] # cm
DIST_SCREEN = 42 # cm

PX_CM = 36.9 # px/cm
BLOCK_ID = [2.28, 2.76, 3.50, 4.23, 5.41, 6.97]
BLOCK_RAD_CM = [7, 5, 3, 1.8, 0.8, 0.27]
BLOCK_RAD_PX = [258, 185, 111, 66, 30, 10]
BLOCK_DIST = 17.5 # cm
FPS_OT = 250 # Hz

#%% Read Video
video, fps, length = read_video(PATH_VIDEO, VID_RES)

#%% Read Data
TRIAL_NO = 3
raw_frames, vid_start, vid_end = read_frames(TRIAL_NO, PATH_FRAMES)
raw_ot = read_optotrak(TRIAL_NO, PATH_OT)
raw_gaze, gaze_xy = read_gaze(vid_start, vid_end, VID_RES, PATH_GAZE)
frames = split_video(video, vid_start, vid_end)

#%% Track Hands + Screen
lbl_fingers, dict_hand, pen_xy = track_hands(frames, True)
dot_xy = track_screen(frames)

#%% Measurements
DOT_RAD = 50 # px
vid_dot_x, vid_dot_y, vid_cm = get_dot_positions(dot_xy, CM_SCREEN, PX_SCREEN, BLOCK_DIST)
gaze_xy = adjust_gaze(gaze_xy, vid_dot_x, vid_dot_y)
hit_dot_pen, hit_dot_gaze = get_hit_states(pen_xy, gaze_xy, vid_dot_x, vid_dot_y, DOT_RAD)

#%% Visualization
frames_out = draw_out_frames(frames, hit_dot_pen, hit_dot_gaze, vid_dot_x, vid_dot_y, DOT_RAD,
                lbl_fingers, dict_hand, pen_xy, gaze_xy, True, 15)
export_video(frames_out, TRIAL_NO, PATH_EXPORT_VIDEO, 15, VID_RES)

#%% Figures
time_ot, time_ah, pen_ot, pen_ah = get_world_position_data(raw_ot, pen_xy, vid_cm, FPS_OT)
export_figure(time_ot, time_ah, pen_ot, pen_ah, PATH_EXPORT_FIG, TRIAL_NO)

#%% Reports
export_report(time_ah, pen_ah, PATH_EXPORT_REPORT, f'fitts_ot_{TRIAL_NO}.csv')
export_report(time_ot, pen_ot, PATH_EXPORT_REPORT, f'fitts_ot_{TRIAL_NO}.csv')
