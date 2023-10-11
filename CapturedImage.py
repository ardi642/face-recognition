
from abc import ABC, abstractmethod

class CapturedImage(ABC):
  
  # bounding box dalam format [x, y, width, height]
  @abstractmethod
  def save(self, identity, bounding_box):
    pass