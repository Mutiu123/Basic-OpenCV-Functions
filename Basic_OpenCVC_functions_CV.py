
import cv2

#load image
img = cv2.imread("Images/image021.jpg")

#**************** convert the image to grayscale ********************
imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Display grayscale imagecd 
cv2.imshow("Grayscale image", imgGray)

#cv2.waitKey(0)
cv2.imwrite("Results/GraySca_image.jpg", imgGray)

