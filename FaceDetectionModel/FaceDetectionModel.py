from abc import ABC, abstractmethod

class FaceDetectionModel(ABC):
  
  # image dalam bentuk gbr
  @abstractmethod
  def detectMultipleFaces(self, image):
    pass