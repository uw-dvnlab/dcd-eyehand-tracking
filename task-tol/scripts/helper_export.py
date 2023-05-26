# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:49:12 2023

@author: zafar
"""
import cv2

def export_video(path_out, filename, frames_out, fps, vid_res):
    """
    Parameters
    ----------
    path_out : string
    filename : string
    frames_out : list of video frames
    fps : int
    vid_res : list of video resolution

    Returns
    -------
    None.
    """
    print('Export Started')
    video = cv2.VideoWriter(
        f'{path_out}/{filename}',
        cv2.VideoWriter_fourcc(*'MP42'),
        fps,
        vid_res)
    for i in range(len(frames_out)):
        print(i)
        video.write(frames_out[i])

    video.release()
    print('Export Complete')
