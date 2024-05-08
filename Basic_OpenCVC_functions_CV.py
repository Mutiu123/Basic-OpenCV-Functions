
import cv2
import numpy as np

#load image
img = cv2.imread("Images/image021.jpg")   #queryImage
cv2.imwrite("Results/RGB_images.jpg", img)

#**************** convert the image to grayscale ********************
# convert RGB image to grayscale
imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Display grayscale imagecd
cv2.imshow("Grayscale image", imgGray)

# Save the reuslt to Results folder
cv2.imwrite("Results/GraySca_image.jpg", imgGray)



#**************** Image Blurring ********************
# Apply Gaussian blur
imgBlurred = cv2.GaussianBlur(img, (11, 11), 0)

# Display the blurred image
cv2.imshow("Blurred image", imgBlurred)

# Save the reuslt
cv2.imwrite("Results/Blurred_image.jpg", imgBlurred)



#**************** Image Resizing ********************
# Resize the image
imgResized = cv2.resize(img, (320, 320))

#Print original image size
print(img.shape)

#Print resized image size
print(imgResized.shape)

# Display the resized image
cv2.imshow("Resized image", imgResized)

# Save the reuslt
cv2.imwrite("Results/Resized_image.jpg", imgResized)


#**************** Image Cropping ********************
# Crop the image
imgCropped = img[180:450,50:450]

# Display the resized image
cv2.imshow("Cropped image", imgCropped)

# Save the reuslt
cv2.imwrite("Results/Cropped_image.jpg", imgCropped)




#**************** Image Rotation ********************
#Get image height and width
(h, w) = img.shape[:2]

#Compute the center of the image
center = (w/2, h/2)

# perform image rotation
mg = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_image = cv2.warpAffine(img, mg, (w, h))

# Display the rotated image
cv2.imshow("Rotated image", rotated_image)

# Save the reuslt
cv2.imwrite("Results/Rotated_image.jpg", rotated_image)




#**************** Image Thresholding ********************
#Set threshold
threshold_lower = 110
threshold_upper = 225

# Apply image thresholding
reth, thresh = cv2.threshold(imgGray, threshold_lower, threshold_upper, cv2.THRESH_BINARY)

# Display the threshold image
cv2.imshow("Thresholded image", thresh)
# Save the reuslt
cv2.imwrite("Results/Thresholded_image.jpg", thresh)




#**************** Edge Detection ********************
# Apply canny edge detection on grayscale image
CamyEdges = cv2.Canny(imgGray, 30, 100)

# Display the edge
cv2.imshow("Detected Edge", CamyEdges)

# Save the reuslt
cv2.imwrite("Results/Edge_detection Image.jpg", CamyEdges)



#**************** Image Filtering ********************
# Create a kernel
Kernel = np.ones((5,5), np.float32)/25

# Apply the kernel to the image
filteredImage = cv2.filter2D(img, -1, Kernel)

# Display the edge
cv2.imshow("Filtered Image", filteredImage)

# Save the reuslt
cv2.imwrite("Results/Filtered_Image.jpg", filteredImage)



#**************** Feature Matching ********************
# Read second image for matching
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





#**************** Image dilation and Erosion ********************
# dilation and erosion are morphological operation that are useful in removing noise such as small
# holes or black sport from binary image

# Apply image thresholding
threshold_lower = 75
threshold_upper = 100


#Define the structuring element
kernel = np.ones((5,5), np.uint8)

# perform dialation on Can
dialated_image = cv2.dilate(CamyEdges, kernel, iterations=1)

# perform erosion
eroded_image = cv2.erode(dialated_image, kernel, iterations=1)

# Display the dialated and eroded image
cv2.imshow("Dialated image", dialated_image)
cv2.imshow("Eroded image", eroded_image)

# Save the reuslt
cv2.imwrite("Results/dialated_image.jpg", dialated_image)
cv2.imwrite("Results/eroded_image.jpg", eroded_image)




#**************** Image Pyramids ********************
# image pyramids fnd application in up or down image scaling, blending,
# reconstruction and texture synthesis.

#create Gaussian pyramid
Mg = img.copy()
mgs = [Mg]
for i in range(8):
    Mg = cv2.pyrDown(Mg)
    mgs.append(Mg)

# Display the image in the Gaussian pyramid
for i in range(4):
    cv2.imshow("Gaussian pyramind Level" + str(i), mgs[i])


#**************** Image Gradients ********************
# Image gradients are useful for the edges image highlighting

#Calculates the x and y gradients using Sobel operator
gradX = cv2.Sobel(imgGray, cv2.CV_64F, 1, 0, ksize=5)
gradY = cv2.Sobel(imgGray, cv2.CV_64F, 1, 0, ksize=5)

# Display the x and y gradients
cv2.imshow("gradients X", gradX)
cv2.imshow("gradients Y", gradY)

# Save the reuslt
cv2.imwrite("Results/X_Gradient_Image.jpg", gradX)
cv2.imwrite("Results/Y_Gradient_Image.jpg", gradY)


#**************** Histogram Equalization ********************
# Histogram equalization is useful for improving the contrast of the images

#Apply histogram equalization
his_equalised = cv2.equalizeHist(imgGray)

# Display equalised image
cv2.imshow("Equalised Image", his_equalised)

# Save the reuslt
cv2.imwrite("Results/Equalised_Image.jpg", his_equalised)


cv2.waitKey(0)
cv2.destroyAllWindows()
