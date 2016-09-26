import numpy as np
import cv2
import sys

def shot(webcamId, webcamId2):
    try:
        cap = cv2.VideoCapture(webcamId)
        cap2 = cv2.VideoCapture(webcamId2)
    except:
        sys.exit('webcam cannot be opened')

    f = open('in.xml', 'w')
    f.write('<?xml version="1.0"?>\n')
    f.write('<opencv_storage>\n')
    f.write('<imagelist>\n')
    n = 1
    while(True):
        ret, frame = cap.read()
        ret2, frame2 = cap2.read()
        cv2.imshow('webcam frame', frame)
        cv2.imshow('webcam2 frame', frame2)
        key = cv2.waitKey(3)&0xFF
        if key == 10:
            # take a freaking picture
            writeleftstr = 'images/left'+str(n)+'.png'
            writerightstr = 'images/right'+str(n)+'.png'
            cv2.imwrite(writeleftstr, frame)
            cv2.imwrite(writerightstr, frame2)
            print('captured '+str(n)+' pair images.')
            f.write('"'+writeleftstr+'"\n')
            f.write('"'+writerightstr+'"\n')
            n += 1
        if key == 27 or key == ord('q'):
            print('bye')
            break

    f.write('</imagelist>\n')
    f.write('</opencv_storage>\n')
    f.close()
    cap.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    shot(1, 2)
