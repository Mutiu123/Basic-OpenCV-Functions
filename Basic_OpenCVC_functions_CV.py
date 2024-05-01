
import cv2
import numpy as np

#load image
img = cv2.imread("Images/image021.jpg")   #queryImage
cv2.imwrite("Results/RGB_images.jpg", img)

#**************** convert the image to grayscale ********************

imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# Display grayscale imagecd
#cv2.imshow("Grayscale image", imgGray)
cv2.imwrite("Results/GraySca_image.jpg", imgGray)


#**************** Image Blurring ********************

# Apply Gaussian blur
imgBlurred = cv2.GaussianBlur(img, (11, 11), 0)

# Display the blurred image
#cv2.imshow("Blurred image", imgBlurred)
# Save the reuslt
cv2.imwrite("Results/Blurred_image.jpg", imgBlurred)



#**************** Image Resizing ********************
# Resize the image
imgResized = cv2.resize(img, (320, 320))

# Display the resized image
#cv2.imshow("Resized image", imgResized)
# Save the reuslt
cv2.imwrite("Results/Resized_image.jpg", imgResized)



#**************** Image Rotation ********************

#Get image height and width
(h, w) = img.shape[:2]
#Compute the center of the image
center = (w/2, h/2)

# perform image rotation
mg = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_image = cv2.warpAffine(img, mg, (w, h))

# Display the rotated image
#cv2.imshow("Rotated image", rotated_image)
# Save the reuslt
cv2.imwrite("Results/Rotated_image.jpg", rotated_image)

#**************** Image Thresholding ********************

# Apply image thresholding
reth, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

# Display the threshold image
#cv2.imshow("Thresholded image", thresh)
# Save the reuslt
cv2.imwrite("Results/Thresholded_image.jpg", thresh)


#**************** Edge Detection ********************

# Apply canny edge detection
edge = cv2.Canny(img, 30, 100)

# Display the edge
#cv2.imshow("Detected Edge", edge)
# Save the reuslt
cv2.imwrite("Results/Edge_detection Image.jpg", edge)

#**************** Image Filtering ********************

# Create a kernel
Kernel = np.ones((5,5), np.float32)/25

# Apply the kernel to the image
filteredImage = cv2.filter2D(img, -0, Kernel)

# Display the edge
cv2.imshow("Filtered Image", filteredImage)
# Save the reuslt
cv2.imwrite("Results/Filtered_Image.jpg", filteredImage)



#**************** Feature Matching ********************
img1 = cv2.imread("Images/image015.jpg")   #matchingImage
# initilise ORB detector
orb = cv2.ORB_create()

# Determine the keypoints and descriptors with ORB
kp1, desc1 = orb.detectAndCompute(img, None)
kp2, desc2 = orb.detectAndCompute(img1, None)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptor
matches = bf.match(desc1,desc2)
#sort in the order of their distance
matches = sorted(matches, key = lambda x:x.distance)

#Draw first 10 matches
img2 = cv2.drawMatches(img, kp1, img1, kp2, matches[:0], None, flags=2)


# Display the edge
cv2.imshow("Feature Matching", img2)
# Save the reuslt
cv2.imwrite("Results/Feature_Matching.jpg", img2)





























































































cv2.waitKey(0)
cv2.destroyAllWindows()
