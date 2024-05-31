import cv2
import numpy as np
#****************** Warp Perspective ***********************
#Perspective Warping involves transforming images to correct perspective distortions caused by cameramisalignment.
#It is crucial for tasks such as camera calibration, augmented reality, panoramic stitching, and document scanning


#load image
img = cv2.imread("Images/imagesL.jpg")

#define image pixels
imageWidth, imageHeight = 275, 182

extractedImagePixel = np.float32([[6,70],[121,54],[55,164],[162,134]])
originalImagePixel = np.float32([[0,0], [imageWidth,0], [0,imageHeight],[imageWidth,imageHeight]])

#compute matrix
matrix = cv2.getPerspectiveTransform(extractedImagePixel,originalImagePixel)
outputImage = cv2.warpPerspective(img,matrix,(imageWidth,imageHeight))

cv2.imshow("ImageL",img)
cv2.imshow("ImageOutput",outputImage)

cv2.waitKey(0)
