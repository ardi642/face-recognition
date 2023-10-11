import os, cv2
from CapturedImage import CapturedImage
from configuration import configuration

from deepface import DeepFace
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["pengenalanWajah"]
collection = db["technosStudio"]

class CapturedTrainingImage(CapturedImage):

  # identity berisi string identitas wajah yang diidentifikasi
  # bounding box  [x, y, width, height]
  # image dalam bentuk rgb perlu diubah ke gbr jika menyimpannya melalui opencv2

  def get_face_embbeding(self, bgr_image):
    embedding = DeepFace.represent(bgr_image, enforce_detection=False)[0]['embedding']
    return embedding

  def save_to_mongodb(self, object):
    collection.insert_one(object)


  def save(self, image, identity, bounding_box):
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    try:
      additional_bottom = configuration['additional_bottom']
    except KeyError:
      additional_bottom = 0
    [x, y, width, height] = bounding_box
    start_point = (x, y)
    end_point = (x + width, y + height + additional_bottom)
    dataset_path = f"DATASET/{identity}/"
    urutan_path = f"URUTAN/{identity}.txt"
    if not os.path.exists(dataset_path):
      os.makedirs(dataset_path)
    
    if not os.path.exists(urutan_path):
      with open(urutan_path, 'w') as file:
        file.write('1')
        urutan = 1
    else:
      with open(urutan_path, 'r+') as file:
        urutan = int(file.read()) + 1
        file.seek(0)
        file.write(str(urutan))

    detected_image = bgr_image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
    file_path = dataset_path + str(urutan) + ".jpg"

    embedding = self.get_face_embbeding(detected_image)
    image_data = {
      "model": "VGG-Face",
      "embedding": embedding,
      "identity": identity
    }
    self.save_to_mongodb(image_data)
    cv2.imwrite(file_path, detected_image)