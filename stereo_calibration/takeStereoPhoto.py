import numpy as np
import cv2
import sys

def shot(webcamId, webcamId2):
    try:
        cap = cv2.VideoCapture(webcamId)
        cap2 = cv2.VideoCapture(webcamId2)
    except:
        sys.exit('webcam cannot be opened')

    n = 1
    while(True):
        ret, frame = cap.read()
        ret2, frame2 = cap2.read()
        cv2.imshow('webcam frame', frame)
        cv2.imshow('webcam2 frame', frame2)
        key = cv2.waitKey(3)&0xFF
        if key == 10:
            # take a freaking picture
            cv2.imwrite('images/left'+str(n)+'.png', frame)
            cv2.imwrite('images/right'+str(n)+'.png', frame2)
            print('captured '+str(n)+' pair images.')
            n += 1
        if key == 27 or key == ord('q'):
            print('bye')
            break

    cap.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('missing 2 argument: webcamId1 webcamId2')
    shot(int(sys.argv[1]), int(sys.argv[2]))
