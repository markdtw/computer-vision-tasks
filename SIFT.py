import cv2
import sys

if len(sys.argv) != 2:
    sys.exit("usage: python2 SIFT.py <picture>");

img = cv2.imread(str(sys.argv[1]))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)

img2 = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('sift_keypoints.png', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
