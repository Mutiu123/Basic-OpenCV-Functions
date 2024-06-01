import cv2
import numpy as np
imag = cv2.imread("Images/imagesL.jpg")
#load image
img = cv2.imread("Images/image021.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# Apply canny edge detection on grayscale image

CamyEdges = cv2.Canny(imgGray, 30, 100)
#
# horizontal_stack = np.hstack((imag,imag))
# vertical_stack = np.vstack((imag,imag))
#
# cv2.imshow("HorizontalStack", horizontal_stack)
# cv2.imshow("VertivalStack", vertical_stack)

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

imagesStack = stack_images(0.9,([imag,imag,imag,imag],[imag,CamyEdges,imag,imag],[imag,imag,imag,imag]))
cv2.imshow("imagesStack", imagesStack)
cv2.waitKey(0)