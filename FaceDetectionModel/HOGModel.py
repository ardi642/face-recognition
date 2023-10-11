from FaceDetectionModel.FaceDetectionModel import FaceDetectionModel
import cv2

class HOGModel(FaceDetectionModel):
  
  def detectMultipleFaces(self, image):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    face_cascade = cv2.CascadeClassifier("HAAR CASCADE DATASET/haarcascade_frontalface_default.xml")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces, weights = hog.detectMultiScale(gray_image, winStride=(2, 2), padding=(10, 10), scale=1.02)
    return faces