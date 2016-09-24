import numpy as np
import cv2
import glob
import sys

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def drawChessboard(height, width):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((height*width,3), np.float32)
    objp[:,:2] = np.mgrid[0:width,0:height].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('images/*.png')

    cnt = 1
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height),None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            print ("Found image", cnt)
            objpoints.append(objp)

            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners)
        
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (width, height), corners, ret)
            cv2.imshow('draw chessboard', img)
            cv2.waitKey(500)
            cnt = cnt + 1

    cv2.destroyAllWindows()

if __name__ == '__main__':
    print ('usage: python calib.py height width (default 7, 9)')
    if len(sys.argv) != 3:
        drawChessboard(7, 9)
    else:
        drawChessboard(int(sys.argv[1]), int(sys.argv[2]))
