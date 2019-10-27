import glob
import os
import numpy as np
import cv2


def read_video_frames_from_dir(frame_dir, size):
    """
    Args:
        frame_dir (str): Directory contains video frames
        size (tuple) : (H , W)
    Returns:
        video (ndArray): (C, T, H, W)
    """
    jpg = os.listdir(frame_dir)
    image = []
    for x in jpg:         #x = 1_0001.jpg ~ 1_0xxx.jpg
        if x.endswith('.jpg'):
            img = cv2.resize(cv2.imread(os.path.join(frame_dir,x)), size)
            image.append(img[:, :, ::-1])
    video =  np.transpose(np.array(image),(3,0,2,1))
    return video
