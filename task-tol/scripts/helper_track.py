# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:43:34 2023

@author: zafar
"""
import mediapipe as mp
import cv2
import numpy as np

def track_hands(frames, gaze_xy, show):
    """
    Parameters
    ----------
    frames : list of video frames
    gaze_xy : array of gaze coordinates
    SHOW : boolean

    Returns
    -------
    lbl_fingers : dictionary of hand landmarks
    dict_hand : dictionary of hand coordinates
    dict_hand_frame : dictionary of tracked frame numbers
    out_frames : list of processed frames
    """
    # init
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils
    out_frames = []

    # create dictionary of hand landmarks
    lbl_fingers = ['thumb', 'index', 'middle', 'ring', 'small']
    dict_hand_key = { 0: 'wrist' }
    for i in range(len(lbl_fingers)):
        for j in range(4):
            dict_hand_key[i*4 + j + 1] = f'{lbl_fingers[i]}_{j}'

    # init dictionary of hand coordinates
    dict_hand = {}
    dict_hand_frame = {}
    for i in range(len(dict_hand_key)):
        dict_hand[dict_hand_key[i]] = []
        dict_hand_frame[dict_hand_key[i]] = []

    for i in range(len(frames)):
        image = frames[i].copy()
        image_rgb = cv2.cvtColor(image,
                                cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # check whether hand is detected
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                for idx, lm in enumerate(hand_lms.landmark):
                    h, w, _ = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # store data frame & coordinates in dictionary
                    dict_hand_frame[dict_hand_key[idx]].append(i)
                    dict_hand[dict_hand_key[idx]].append([cx,cy])

                if show: # draw hands
                    mp_draw.draw_landmarks(image,
                                          hand_lms,
                                          mp_hands.HAND_CONNECTIONS,
                                          mp_draw.DrawingSpec(color=(245,117,66),
                                                             thickness=4,
                                                             circle_radius=2),
                                          mp_draw.DrawingSpec(color=(245,66,230),
                                                             thickness=4,
                                                             circle_radius=2))

        if show: # draw gaze
            image = draw_gaze(image,
                              i,
                              gaze_xy)
            cv2.imshow("Output",
                       image)
            cv2.waitKey(1)

        # save frames for export
        out_frames.append(image)

    for key in dict_hand:
        dict_hand[key] = np.array(dict_hand[key])

    return lbl_fingers, dict_hand, dict_hand_frame, out_frames

def draw_gaze(frame, f_no, gaze_xy):
    """
    Parameters
    ----------
    frame : numpy image array
    f_no : int
    gaze_xy : array of gaze coordinates

    Returns
    -------
    frame : numpy image array
    """
    cv2.circle(frame,
               tuple(gaze_xy[f_no,:]),
               8,
               (0, 0, 255),
               4)

    return frame
