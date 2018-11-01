import cv2
import numpy as np
import sys


#detect.py should be in the same forlder as images
#if you need blur add -b after image file

filename = sys.argv[1]

try:
    if sys.argv[2]=='-b':  do_blur=1
except IndexError:
    do_blur=0   

img = cv2.imread(filename)

def im_size_gray(imge, sz=600): 
    #resize for speed and overview
    r = 600.0 / imge.shape[1]
    dim = (600, int(imge.shape[0] * r))
    imge = cv2.resize(imge, dim)

    #monochrome, blur
    im = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)
    im = cv2.fastNlMeansDenoising(im, h=10) 
    im = cv2.GaussianBlur(im, (9,9), 0)
    return im


fim = im_size_gray(img, 600)

#min-max = black white 
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(fim)


### Flare1 by white area
# threshold the image to reveal light regions in the blurred image
mask = cv2.threshold(fim, maxVal-7, 255, cv2.THRESH_BINARY)[1]
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=4)

#calcul % of white (flare) area
scr1 = round( float(cv2.countNonZero(mask))/(mask.shape[0]*mask.shape[1]), 2 )
flare1=1 if scr1>0.15 else 0

### Flare2  by ~white variation >5000 <8500
msk2 = cv2.threshold(fim, 250, 255, cv2.THRESH_BINARY)[1]
scr2 = round( msk2.var(), 0)
flare2=1 if (scr2>5000)&(scr2<8500)  else 0

### Flare3 by absence of black 
flare3=1 if minVal > 40  else 0

'''
### Flare4 by present of white oval blops
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
#by color might not work
params.blobColor = 255
# Change thresholds
params.minThreshold = 3;
params.maxThreshold = 255;
# Filter by Area.
params.filterByArea = True
params.minArea = 200
# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.3
# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)
# Detect blobs.
keypoints = detector.detect(im)
'''


#print("1({}):{}, 2({}):{}, 3:{}".format(scr1, flare1, scr2, flare2, flare3))

print( round( flare1*0.35+flare2*0.45+flare3*0.6))

if do_blur==1:
    #calculating blur with Laplacian variation
    bim = cv2.Laplacian(fim, cv2.CV_64F)
    blur=1 if np.var(bim) < 100 else 0
    print("blur=", blur)
   
