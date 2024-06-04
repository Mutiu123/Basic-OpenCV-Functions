import cv2
import numpy as np

def NullVal(n):
    pass

def stack_images(scale,imageArray):
    rows = len(imageArray)
    colums = len(imageArray[0])
    rowsAvailable = isinstance(imageArray[0], list)
    width = imageArray[0][0].shape[1]
    height = imageArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, colums):
                if imageArray[x][y].shape[:2] == imageArray[0][0].shape[:2]:
                    imageArray[x][y] = cv2.resize(imageArray[x][y], (0, 0), None, scale, scale)
                else:
                    imageArray[x][y] = cv2.resize(imageArray[x][y], (imageArray[0][0].shape[1], imageArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imageArray[x][y].shape) == 2: imageArray[x][y] = cv2.cvtColor(imageArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        horizontal = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            horizontal[x] = np.hstack(imageArray[x])
        value = np.vstack(horizontal)
    else:
        for x in range(0, rows):
            if imageArray[x].shape[:2] == imageArray[0].shape[:2]:
                imageArray[x] = cv2.resize(imageArray[x], (0, 0), None, scale, scale)
            else:
                imageArray[x] = cv2.resize(imageArray[x], (imageArray[0].shape[1], imageArray[0].shape[0]), None, scale,
                                         scale)
            if len(imageArray[x].shape) == 2: imageArray[x] = cv2.cvtColor(imageArray[x], cv2.COLOR_GRAY2BGR)
        horizontal = np.hstack(imageArray)
        value = horizontal
    return value

cv2.namedWindow("TrackCol")
cv2.resizeWindow("TrackCol", 620,230)
cv2.createTrackbar("Mini Hue", "TrackCol", 0, 179, NullVal)
cv2.createTrackbar("Max Hue", "TrackCol", 21,179, NullVal)
cv2.createTrackbar("Mini Sat", "TrackCol", 110,255, NullVal)
cv2.createTrackbar("Max Sat", "TrackCol", 240,255, NullVal)
cv2.createTrackbar("Mini Val", "TrackCol", 152,255, NullVal)
cv2.createTrackbar("Max Val", "TrackCol", 255,255, NullVal)


while (1):
    imag = cv2.imread("Images/box.jpeg")
    HSV_image = cv2.cvtColor(imag, cv2.COLOR_BGR2HSV)
    Mini_hue = cv2.getTrackbarPos("Mini Hue", "TrackCol")
    Max_hue = cv2.getTrackbarPos("Max Hue", "TrackCol")
    Mini_Sat = cv2.getTrackbarPos("Mini Sat", "TrackCol")
    Max_Sat = cv2.getTrackbarPos("Max Sat", "TrackCol")
    Mini_Val = cv2.getTrackbarPos("Mini Val", "TrackCol")
    Max_Val = cv2.getTrackbarPos("Max Val", "TrackCol")
    print( Mini_hue, Max_hue,  Mini_Sat, Max_Sat, Mini_Val, Max_Val)

    L_band = np.array([Mini_hue, Mini_Sat, Mini_Val])
    Up_band = np.array([Max_hue, Max_Sat, Max_Val])
    mask_image = cv2.inRange(HSV_image, L_band, Up_band)
    Final_image = cv2.bitwise_and(imag, imag, mask=mask_image)

    stackImage = stack_images(0.7, ([imag, HSV_image],[mask_image, Final_image]))
    cv2.imshow("Stacked image", stackImage )

    #Stack_image = st

    cv2.waitKey(1)