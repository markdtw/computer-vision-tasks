from matplotlib import pyplot as plt
import numpy as np
import cv2
import glob
import sys

# These 2 lists are for later usage after chessboard found.
l_goodpair = []
r_goodpair = []

# The termination criteria and flags are checked out from opencv sample (in calling order).
chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
cornersub_criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 30, 0.01)
stereocalib_criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
stereocalib_flags = cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_SAME_FOCAL_LENGTH + \
        cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5

def numDispsCallBack(x):
    pass
def bSizeCallBack(x):
    pass

def stereoRectificationProcess(rectify_scale, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F):
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify( \
            cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (640, 480), R, T, alpha=rectify_scale)
    l_maps = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (640, 480), cv2.CV_16SC2)
    r_maps = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (640, 480), cv2.CV_16SC2)

    
    for l, r in zip(l_goodpair, r_goodpair):
        l_gray = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
        r_gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        l_imgremap = cv2.remap(l_gray, l_maps[0], l_maps[1], cv2.INTER_LINEAR)
        r_imgremap = cv2.remap(r_gray, r_maps[0], r_maps[1], cv2.INTER_LINEAR)

        cv2.imshow('left image_remap', l_imgremap)
        cv2.imshow('right image_remap', r_imgremap)
        cv2.waitKey(500)

    cv2.destroyAllWindows()

    """
    lcap = cv2.VideoCapture(1)
    rcap = cv2.VideoCapture(2)
    cv2.namedWindow('disparity_frame')
    cv2.createTrackbar('numDisparities', 'disparity_frame', 1, 20, numDispsCallBack)
    cv2.createTrackbar('blockSize', 'disparity_frame', 7, 30, bSizeCallBack)
    
    while(True):
        
        lret, lframe = lcap.read()
        rret, rframe = rcap.read()
        
        lframe_remap = cv2.remap(lframe, l_maps[0], l_maps[1], cv2.INTER_LINEAR)
        rframe_remap = cv2.remap(rframe, r_maps[0], r_maps[1], cv2.INTER_LINEAR)
        lremap_gray = cv2.cvtColor(lframe_remap, cv2.COLOR_BGR2GRAY)
        rremap_gray = cv2.cvtColor(rframe_remap, cv2.COLOR_BGR2GRAY)

        numDisparitiesCallBack = cv2.getTrackbarPos('numDisparities', 'disparity_frame')
        blockSizeCallBack = cv2.getTrackbarPos('blockSize', 'disparity_frame')
        blockSizeCallBack = blockSizeCallBack > 1 and blockSizeCallBack or 2
        
        stereo = cv2.StereoBM_create(numDisparities=numDisparitiesCallBack*16, blockSize=blockSizeCallBack*2+1)
        disparity = stereo.compute(lremap_gray, rremap_gray)
        cv2.imshow('disparity_frame', disparity)

        cv2.imshow('left_webcam remap', lframe_remap)
        cv2.imshow('right_webcam remap', rframe_remap)
        key = cv2.waitKey(5)&0xFF
        if key == 27 or key == ord('q'):
            print('bye')
            break

    lcap.release()
    rcap.release()
    cv2.destroyAllWindows()
    """
    
def getCalibratefromStereoImage(objpoints, l_imgpoints, r_imgpoints):
    
    # from OpenCV docs: if any of CV_CALIB_FIX_ASPECT_RATIO... are specified, the matrix components must be initialized.
    cameraMatrix1 = cv2.initCameraMatrix2D(objpoints, l_imgpoints, (640, 480), 0);
    cameraMatrix2 = cv2.initCameraMatrix2D(objpoints, r_imgpoints, (640, 480), 0);
    distCoeffs1 = None
    distCoeffs2 = None

    # directly call stereCalibrate from OpenCV library
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
            cv2.stereoCalibrate(objpoints, l_imgpoints, r_imgpoints, \
            cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (640, 480), \
            criteria=stereocalib_criteria, flags=stereocalib_flags)

    return retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F

def drawChessboard(height, width):
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((height*width,3), np.float32)
    objp[:,:2] = np.mgrid[0:width,0:height].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space

    l_imgpoints = [] # 2d points in left image plane.
    r_imgpoints = [] # 2d points in right image plane.
    
    l_images = glob.glob('images_treasure/left*.png')
    r_images = glob.glob('images_treasure/right*.png')

    for cnt in range(1, len(l_images)+1):
        l_img = cv2.imread('images/left'+str(cnt)+'.png')
        r_img = cv2.imread('images/right'+str(cnt)+'.png')

        l_gray = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
        r_gray = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)

        # First: find chessboard corners by cv2.findChessboardCorners.
        l_ret, l_corners = cv2.findChessboardCorners(l_gray, (width, height), chessboard_flags)
        r_ret, r_corners = cv2.findChessboardCorners(r_gray, (width, height), chessboard_flags)

        if l_ret and r_ret:
            print ("Found image pair:", cnt)
            l_goodpair.append(l_img)
            r_goodpair.append(r_img)
            
            # Second: find subpixel coordinates by cv2.cornerSubPix.
            cv2.cornerSubPix(l_gray, l_corners, (11, 11), (-1, -1), cornersub_criteria)
            cv2.cornerSubPix(r_gray, r_corners, (11, 11), (-1, -1), cornersub_criteria)
            
            # Third: store keypoints.
            l_imgpoints.append(l_corners)
            r_imgpoints.append(r_corners)
            objpoints.append(objp)

            """
            # Draw and display the corners
            cv2.drawChessboardCorners(l_img, (width, height), l_corners, l_ret)
            cv2.drawChessboardCorners(r_img, (width, height), r_corners, r_ret)
            cv2.imshow('left chessboard', l_img)
            cv2.imshow('right chessboard', r_img)
            cv2.waitKey(300)
            """            

    cv2.destroyAllWindows()
    return l_imgpoints, r_imgpoints, objpoints

if __name__ == '__main__':
    print ('usage: python calib.py height width (default 7, 10)')
    # Step 1: For each stereo pair we need to find the chessboard and store the keypoints.
    if len(sys.argv) == 2:
        rectify_scale = float(sys.argv[1])
    else:
        rectify_scale = 1
    
    l_imgpoints, r_imgpoints, objpoints = drawChessboard(7, 10)

    print ('calibrating...')
    # Step 2: Compute calibration.
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
            getCalibratefromStereoImage(objpoints, l_imgpoints, r_imgpoints)

    # Step 2.5 TODO: Save the calibration stats to disk for future use

    print ('rectifying...')
    # Step 3: Stereo rectification
    stereoRectificationProcess(rectify_scale, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F)
