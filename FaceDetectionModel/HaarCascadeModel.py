from FaceDetectionModel.FaceDetectionModel import FaceDetectionModel
import cv2

class HaarCascadeModel(FaceDetectionModel):
  
  def detectMultipleFaces(self, image):
    face_cascade = cv2.CascadeClassifier("HAAR CASCADE DATASET/haarcascade_frontalface_default.xml")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=9)
    return faces