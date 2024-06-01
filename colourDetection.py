import cv2
import numpy as np
def NullVal():
    pass
cv2.namedWindow("TrackCol")
cv2.resizeWindow("TrackCol", 620,230)
cv2.createTrackbar("TrackCol", "TrackCol", 0, 179, NullVal)
cv2.createTrackbar("Mini Hue", "TrackCol", 179,179, NullVal)
cv2.createTrackbar("Max Hue", "TrackCol", 0,255, NullVal)
cv2.createTrackbar("Mini Sat", "TrackCol", 0,255, NullVal)
cv2.createTrackbar("Max Hue", "TrackCol", 0,255, NullVal)
cv2.createTrackbar("Mini Sat", "TrackCol", 0,255, NullVal)

#cv2.createTrackbar("Mini Hue", "TrackCol", 0, 179, NullVal)
imag = cv2.imread("Images/box.jpeg")

HSVimage = cv2.cvtColor(imag, cv2.COLOR_BGR2HSV)


cv2.imshow("originalImage", imag)
cv2.imshow("HSVmage", HSVimage)


cv2.waitKey(0)