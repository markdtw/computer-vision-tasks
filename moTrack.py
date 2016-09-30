import cv2
import numpy as np

def searchForMovement(threshold_image, camera_feed, threshold=500):
    temp = threshold_image
    contours, hierarchy = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areaArray = []

    if len(contours) > 0:
        # compute areas, find the largest contour in the image
        areas = [cv2.contourArea(c) for c in contours]
        for i in range(len(areas)):
            if areas[i] < threshold:
                continue
            x, y, w, h = cv2.boundingRect(contours[i])
            cv2.rectangle(camera_feed, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return camera_feed
        
def trackMotion(frame1, frame2, sensitivity_value=20, blur_size=10, kernel=(15,15)):
    
    # convert frames into gray scale images
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # perfrom frame differencing to get "intensity image"
    frame_difference = cv2.absdiff(gray_frame1, gray_frame2)

    # threshol intensity image at a given sensitivity value
    retval, threshold_image = cv2.threshold(frame_difference, sensitivity_value, 255, cv2.THRESH_BINARY)
    # remove noise
    threshold_image = cv2.blur(threshold_image, (blur_size, blur_size))
    # threshold again
    retval, threshold_image = cv2.threshold(frame_difference, sensitivity_value, 255, cv2.THRESH_BINARY)

    # might be useful
    # threshold_image = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, kernel)
    bounded_image = searchForMovement(threshold_image, frame1)

    return bounded_image

if __name__ == '__main__' :
    import sys
    print('Perform motion tracking test...')
    cap = cv2.VideoCapture('v1.mp4')
    cnt = 0

    if not cap.isOpened():
        print('fail to read video')
        sys.exit(0)

    ## read first frame for frame differencing
    ret, frame2 = cap.read()
    while(cap.isOpened):
        frame1 = frame2
        ret, frame2 = cap.read()
        if not ret:
            break
        result = trackMotion(frame1, frame2)
        cv2.imshow("display", result)
        cv2.waitKey(1)

    cap.release()
