
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



#**************** Image Resizing ********************
imgResized = cv2.resize(img, (320, 320))

# Display the resized image
cv2.imshow("Resized image", imgResized)
# Save the reuslt
cv2.imwrite("Results/Resized_image.jpg", imgResized)



#**************** Image Rotation ********************

#Get image height and width
(h, w) = img.shape[:2]
#Compute the center of the image
center = (w/2, h/2)

#perform rotation
mg = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_image = cv2.warpAffine(img, mg, (w, h))

# Display the rotated image
cv2.imshow("Rotated image", rotated_image)
# Save the reuslt
cv2.imwrite("Results/Rotated_image.jpg", rotated_image)





































































































cv2.waitKey(0)
cv2.destroyAllWindows()
