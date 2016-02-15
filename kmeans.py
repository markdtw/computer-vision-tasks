import numpy as np
import cv2
import sys

if len(sys.argv) != 2:
    sys.exit("usage: python2 kmeans.py <picture>");

img = cv2.imread(str(sys.argv[1]))
Z = img.reshape((-1, 3))

Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = int(sys.argv[2])
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imshow('k-means', res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
