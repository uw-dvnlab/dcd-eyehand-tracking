# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 19:48:56 2023

@author: zafar
"""
import cv2

# GAZE
def draw_gaze(frame, fNo, gaze_xy):
    N_TRAIL = 1 # frames
    for n in range(N_TRAIL):
        if fNo-n > 0:
            cv2.circle(frame, tuple(gaze_xy[fNo-(N_TRAIL - n),:]),
                       8, (0, 0, 255), 4)
    
    return frame

# HAND
def draw_hand(frame, fNo, lbl_fingers, dict_hand, pen_xy):
    # Lines
    for finger in lbl_fingers:
        for seg in range(4):
            if seg==0: # wrist
                xy_0 = dict_hand['wrist'][fNo,:]
                xy_1 = dict_hand[f'{finger}_0'][fNo,:]
            else:
                xy_0 = dict_hand[f'{finger}_{seg-1}'][fNo,:]
                xy_1 = dict_hand[f'{finger}_{seg}'][fNo,:]
            cv2.line(frame, tuple(xy_0), tuple(xy_1), (50, 50, 50), 2)
    # Points
    for key in dict_hand:
        if key=='wrist':
            cv2.circle(frame, tuple(dict_hand[key][fNo,:]),
                       10, (0, 255, 0), -1)        
        else:
            cv2.circle(frame, tuple(dict_hand[key][fNo,:]),
                       2, (255, 0, 0), -1)
    # PEN
    cv2.circle(frame, tuple(pen_xy[fNo,:]),
                10, (0, 255, 0), -1)
    
    return frame

# DOTS
def draw_dots(frame, fNo, hit_dot_pen, hit_dot_gaze, vid_dot_x, vid_dot_y, dot_rad):
    for d in range(2):
        if hit_dot_pen[d][fNo]:
            cv2.circle(frame, (int(vid_dot_x[d][fNo]), int(vid_dot_y[d][fNo])),
                       dot_rad, (0, 255, 0), 3)
        elif hit_dot_gaze[d][fNo]:
            cv2.circle(frame, (int(vid_dot_x[d][fNo]), int(vid_dot_y[d][fNo])),
                       dot_rad, (0, 0, 255), 3)
        else:
            cv2.circle(frame, (int(vid_dot_x[d][fNo]), int(vid_dot_y[d][fNo])),
                       dot_rad, (0, 0, 0), 2)
    
    return frame

def draw_out_frames(frames, hit_dot_pen, hit_dot_gaze, vid_dot_x, vid_dot_y, dot_rad,
                    lbl_fingers, dict_hand, pen_xy, gaze_xy, show, dt):
    frames_out = []
    for fNo in range(len(frames)-1):
        frame = frames[fNo].copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = draw_dots(frame, fNo, hit_dot_pen, hit_dot_gaze, vid_dot_x, vid_dot_y, dot_rad)
        frame = draw_hand(frame, fNo, lbl_fingers, dict_hand, pen_xy)
        frame = draw_gaze(frame, fNo, gaze_xy)
        
        frames_out.append(frame)
        
        if show:
            cv2.imshow("", frame)
            cv2.waitKey(dt)
            
    return frames_out
