import cv2
import numpy as np

#Draw image
img = np.zeros((512,512,3),np.uint8)

# Add text to the image
Scale = 1.5
Thickness = 2
cv2.putText(img, "SAMPLE", (200,40), cv2.FONT_HERSHEY_TRIPLEX,Scale,(0,0,255),Thickness)


# Add line to the image
imageHeight = img.shape[0]
imageWidth = img.shape[1]

cv2.line(img,(0,0), (imageWidth,imageHeight), (255,255,0),Thickness)

# Add rectangle to the image
cv2.rectangle(img, (150, 350), (300, 400), (0,255,0),cv2.FILLED)

# Add circuit to the image
CirRadius = 60
cv2.circle(img, (320, 140), CirRadius, (0,255,255),Thickness)



cv2.imshow("Image", img)
#print(img)

cv2.waitKey(0)