import tkinter as tk
from TopLevel.WebcamTopLevel import WebcamTopLevel
from TopLevel.FaceRecognitionTopLevel import FaceRecognitionTopLevel

from CapturedTrainingImage import CapturedTrainingImage

class DatasetFrame(tk.Frame):
  def __init__(self, master = None, **kwargs):
    super().__init__(master = None, **kwargs, pady=40, padx=40)
    self.master = master

    self.training_button = tk.Button(self, text="kumpul data training", command=self.show_webcam_training)
    self.training_button.pack(side="left", padx=(0, 20))

    self.recognition_button = tk.Button(self, text="Uji pengenalan wajah", command=self.show_face_recognition_webcam)
    self.recognition_button.pack(side="left")

  def show_webcam_training(self):
    captured_images = [CapturedTrainingImage()]
    dataset_status = "training"
    webcam_window = WebcamTopLevel(self, captured_images=captured_images, dataset_status=dataset_status)

  def show_face_recognition_webcam(self):
    face_recognition_webcam_window = FaceRecognitionTopLevel(self)