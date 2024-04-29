
import cv2

#load image
img = cv2.imread("Images/image021.jpg")
cv2.imwrite("Results/RGB_images.jpg", img)

#**************** convert the image to grayscale ********************
imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# Display grayscale imagecd
cv2.imshow("Grayscale image", imgGray)
cv2.imwrite("Results/GraySca_image.jpg", imgGray)


#**************** Image Blurring ********************
imgBlurred = cv2.GaussianBlur(img, (11, 11), 0)

# Display the blurred image
cv2.imshow("Blurred image", imgBlurred)
# Save the reuslt
cv2.imwrite("Results/Blurred_image.jpg", imgBlurred)




































































































cv2.waitKey(0)
cv2.destroyAllWindows()
