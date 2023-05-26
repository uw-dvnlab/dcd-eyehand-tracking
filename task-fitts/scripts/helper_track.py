# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 20:02:03 2023

@author: zafar
"""
import mediapipe as mp
import cv2
import numpy as np

def track_hands(frames, show):
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    
    lbl_fingers = ['thumb', 'index', 'middle', 'ring', 'small']
    dict_hand_key = {
        0: 'wrist'}
    for i in range(len(lbl_fingers)):
        for j in range(4):
            dict_hand_key[i*4 + j + 1] = f'{lbl_fingers[i]}_{j}'
    
    dict_hand = {}
    for i in range(len(dict_hand_key)):
        dict_hand[dict_hand_key[i]] = []
    
    for i in range(len(frames)):
        image = frames[i].copy()
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)
        
        # checking whether a hand is detected
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks: # working with each hand
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    dict_hand[dict_hand_key[id]].append([cx,cy])
                
                if show:
                    mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
        
        if show:
            cv2.imshow("Output", image)
            cv2.waitKey(1)
    
    for key in dict_hand:
        dict_hand[key] = np.array(dict_hand[key])
                
    pen_xy = (0.5 * (dict_hand['thumb_3'] + dict_hand['index_3'])).astype(int)
    
    return lbl_fingers, dict_hand, pen_xy

def track_screen(frames):
    # initialize OpenCV's special multi-object tracker
    trackers = cv2.legacy.MultiTracker_create()
    # loop over frames from the video stream
    box_xy = [[], []]
    # while True:
    for fNo in range(len(frames)):
        # grab the current frame
        frame = frames[fNo].copy()
        # grab the updated bounding box coordinates (if any) for each
        # object that is being tracked
        (success, boxes) = trackers.update(frame)
        for i in range(len(boxes)):
            (x, y, w, h) = [int(v) for v in boxes[i]]
            box_xy[i].append([int(x+w/2), int(y+h/2)])
    
        # loop over the bounding boxes and draw then on the frame
        for i in range(len(box_xy)):
            if fNo>0:
                box = box_xy[i][-1]
                cv2.circle(frame,(box[0],box[1]),5,(0, 0, 255),-1)
    
        # show the output frame
        cv2.imshow("Choose left then right screen corners",
                   frame)
        key = cv2.waitKey(1) & 0xFF
    
        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
        if fNo == 0:
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            for i in range(len(box_xy)):
                boxSel = cv2.selectROI("Choose left then right screen corners",
                                       frame,
                                       fromCenter = False,
                                       showCrosshair = True)
                (x, y, w, h) = [int(v) for v in boxSel]
                box_xy[i].append([int(x+w/2), int(y+h/2)])
                # create a new object tracker for the bounding box and add it
                # to our multi-object tracker
                tracker = cv2.legacy.TrackerCSRT_create()
                trackers.add(tracker, frame, boxSel)
    
        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            break
    
    # close all windows
    cv2.destroyAllWindows()
    
    dot_xy = [np.asarray(box_xy[0]), np.asarray(box_xy[1])]
    
    return dot_xy

def get_dot_positions(dot_xy, cm_screen, px_screen, block_dist):
    vid_px_corners = np.mean(np.abs(dot_xy[0][:,0] - dot_xy[1][:,0])) # vid px
    vid_px_center = 0.55*dot_xy[0][:,0] + 0.45*dot_xy[1][:,0] # vid px
    cm_corners = cm_screen[0] # cm
    vid_cm = vid_px_corners / cm_corners

    vid_dot_x = [vid_px_center - (block_dist/2) * vid_cm,
                 vid_px_center + (block_dist/2) * vid_cm]
    vid_dot_y = [dot_xy[0][:,1] - 220,
                 dot_xy[1][:,1] - 220]
    
    return vid_dot_x, vid_dot_y, vid_cm

def get_hit_states(pen_xy, gaze_xy, vid_dot_x, vid_dot_y, dot_rad):
    hit_dot_pen = [[False], [False]]
    hit_dot_gaze = [[False], [False]]
    for i in range(1, len(pen_xy[:,0])):
        # get pen distances to dots
        dist_dot_pen = [
            np.sqrt((pen_xy[i-1,0] - vid_dot_x[0][i])**2
                         + 0 * (pen_xy[i-1,1] - vid_dot_y[0][i] - 25)**2),
            np.sqrt((pen_xy[i-1,0] - vid_dot_x[1][i])**2
                         + 0 * (pen_xy[i-1,1] - vid_dot_y[1][i] - 25)**2)
            ]
        
        for d in range(2):
            if dist_dot_pen[d] <= 1.5*dot_rad:
                hit_dot_pen[d].append(True)
            else:
                hit_dot_pen[d].append(False)
                
        # get pen distances to dots
        dist_dot_gaze = [
            np.sqrt((gaze_xy[i-1,0] - vid_dot_x[0][i])**2
                         + (gaze_xy[i-1,1] - vid_dot_y[0][i])**2),
            np.sqrt((gaze_xy[i-1,0] - vid_dot_x[1][i])**2
                         + (gaze_xy[i-1,1] - vid_dot_y[1][i])**2)
            ]
        
        for d in range(2):
            if dist_dot_gaze[d] <= dot_rad + 75:
                hit_dot_gaze[d].append(True)
            else:
                hit_dot_gaze[d].append(False)
                
    return hit_dot_pen, hit_dot_gaze

def adjust_gaze(gaze_xy, vid_dot_x, vid_dot_y):
    cx = 0.5*(vid_dot_x[0] + vid_dot_x[1])
    dx = gaze_xy[0,0] - cx[0]
    gaze_xy[:,0] -= int(dx)
    
    cy = 0.5*(vid_dot_y[0] + vid_dot_y[1])
    dy = gaze_xy[0,1] - cy[0]
    gaze_xy[:,1] -= int(dy)
    
    return gaze_xy

def get_world_position_data(raw_ot, pen_xy, vid_cm, fps_ot):
    init_ot = 0.6 # seconds
    wrist_ot = raw_ot['finger x'].values # mm
    wrist_ot = wrist_ot[int(init_ot*fps_ot):]
    wrist_ot -= wrist_ot[0]
    time_ot = np.array(list(range(len(wrist_ot)))) / fps_ot

    fps_ah = 30
    wrist_ah = pen_xy[:,0].astype(float) #dict_hand['wrist'][:,0].astype(float) # px
    wrist_ah -= wrist_ah[0]
    time_ah = np.array(list(range(len(wrist_ah)))) / fps_ah
    # convert pixels to mm
    wrist_ah *= 10/vid_cm
    
    return time_ot, time_ah, wrist_ot, wrist_ah