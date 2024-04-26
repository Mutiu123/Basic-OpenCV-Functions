import cv2
print("Package Imported")

"""""
#********************* IMPORT IMAGE ********************

img = cv2.imread("Images/lenna.png")

cv2.imshow("Image", img)
cv2.waitKey(0)

#********************* READ VIDEO ********************

VideoCap = cv2.VideoCapture("Videos/dog.mp4")
while True:
    successful, img = VideoCap.read()
    cv2.imshow("video", img)

    if cv2.waitKey(1) & 0XFF ==ord('q'):
        break

"""""
#********************* READ WEBCAM ********************

WebcamVideo = cv2.VideoCapture(0)
WebcamVideo.set(1,576)
WebcamVideo.set(2,576)
WebcamVideo.set(10,100)

while True:
    successful, image = WebcamVideo.read()
    cv2.imshow("webcamvideos", image)

    if cv2.waitKey(1) & 0XFF ==ord('q'):
        break