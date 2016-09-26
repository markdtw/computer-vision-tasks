#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

if __name__ == '__main__':
    print('loading images...')
    imgL = cv2.imread('images/left1.png')  # downscale images for faster processing
    imgR = cv2.imread('images/right1.png')

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    # disparity range is tuned for 'aloe' image pair
    """
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )
    """

    print('computing disparity...')
    #disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    l_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    r_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    disp = stereo.compute(l_gray, r_gray)

    cv2.imshow('left', imgL)
    #cv2.imshow('disparity', (disp-min_disp)/num_disp)
    cv2.imshow('disparity', disp)
    cv2.waitKey()
    cv2.destroyAllWindows()
