from FaceDetectionModel.FaceDetectionModel import FaceDetectionModel
from mtcnn.mtcnn import MTCNN
import cv2
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
class MTCNNModel(FaceDetectionModel):
  
  def __init__(self):
    self.detector = MTCNN()
  def detectMultipleFaces(self, image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_results = self.detector.detect_faces(rgb_image)
    face_length = len(face_results)
    faces = []
    if face_length > 0:
      for n in range(face_length):
        faces.append(face_results[n]['box'])
    
    return faces  