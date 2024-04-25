import cv2
print("Package Imported")


img = cv2.imread("Images/lenna.png")

cv2.imshow("Image", img)
cv2.waitKey(0)
