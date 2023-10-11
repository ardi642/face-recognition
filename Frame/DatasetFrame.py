import tkinter as tk
from tkinter import messagebox
from TopLevel.WebcamTopLevel import WebcamTopLevel
from TopLevel.FaceRecognitionTopLevel import FaceRecognitionTopLevel

from CapturedTrainingImage import CapturedTrainingImage
import pymongo

from sklearn.svm import SVC
import pickle

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["pengenalanWajah"]
collection = db["technosStudio"]

class DatasetFrame(tk.Frame):
  def __init__(self, master = None, **kwargs):
    super().__init__(master = None, **kwargs, pady=40, padx=40)
    self.master = master

    self.collecttion_button = tk.Button(self, text="kumpul data training", command=self.show_webcam_training)
    self.collecttion_button.pack(side="left", padx=(0, 20))

    self.training_button = tk.Button(self, text="latih model", command=self.train_model)
    self.training_button.pack(side="left", padx=(0, 20))

    self.recognition_button = tk.Button(self, text="Uji pengenalan wajah", command=self.show_face_recognition_webcam)
    self.recognition_button.pack(side="left")

  def show_webcam_training(self):
    captured_images = [CapturedTrainingImage()]
    dataset_status = "training"
    webcam_window = WebcamTopLevel(self, captured_images=captured_images, dataset_status=dataset_status)

  def show_face_recognition_webcam(self):
    face_recognition_webcam_window = FaceRecognitionTopLevel(self)

  def train_model(self):
    embeddings = []
    identities = []
    for data in collection.find({}, {'_id': 0, 'embedding': 1, 'identity': 1}):
      embeddings.append(data['embedding'])
      identities.append(data['identity'])
    
    svm = SVC(probability=True)
    svm.fit(embeddings, identities)
    with open('SVM_Model/svm_model.pkl','wb') as model_file:
        pickle.dump(svm, model_file)

    messagebox.showinfo(
      title='berhasil',
      message=f"model pengenalan wajah berhasil dilatih"
    )

    
  