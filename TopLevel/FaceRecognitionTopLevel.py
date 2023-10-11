import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import os
from FaceDetectionModel.MTCNNModel import MTCNNModel
from FaceDetectionModel.HaarCascadeModel import HaarCascadeModel
from FaceDetectionModel.HOGModel import HOGModel

from deepface import DeepFace
import logging

from configuration import configuration

import logging

root_logger = logging.getLogger()
root_logger.propagate = False  # Disable propagation

detection_model_options = ['haar cascade', 'mtcnn', 'hog']

class FaceRecognitionTopLevel(tk.Toplevel):
  def __init__(self, master, **kwargs):
    super().__init__(master, **kwargs)
    self.geometry("1000x700")
    
    self.title("uji pengenalan wajah")
    self.cap = cv2.VideoCapture(0)
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    self.webcam_label = tk.Label(self)
    self.webcam_label.pack(fill="both", expand=True, side="top")

    self.bottom_frame = tk.Frame(self)
    self.bottom_frame.pack(fill="y", side="top", pady=(0, 25))

    self.selected_model = tk.StringVar()
    self.selected_model.trace("w", self.update_detector)

    self.detection_model_label = tk.Label(self.bottom_frame, text='model pendeteksi wajah')
    self.detection_model_label.grid(column=0, row=1, padx=5, pady=5)
    self.detection_model_options = ttk.OptionMenu(self.bottom_frame, self.selected_model,
                                                  detection_model_options[0], *detection_model_options)
    self.detection_model_options.grid(column=1, row=1, padx=5, pady=5)

    self.threshold_label = tk.Label(self.bottom_frame, text='threshold maksimal cosine similarity')
    self.threshold_label.grid(column=0, row=2, padx=5, pady=5)

    self.threshold = tk.DoubleVar()
    configuration['max_threshold'] = 0.1
    self.threshold.set(0.1)

    self.threshold_entry = tk.Entry(self.bottom_frame, textvariable=self.threshold)
    self.threshold_entry.grid(column=1, row=2, pady=5, padx=5)

    self.threshold.trace("w", self.update_threshold)

    self.initialize_face_detector()

    self.show_webcam()

    self.protocol("WM_DELETE_WINDOW", self.process_destroy)

  def update_threshold(self, *args):
    try:
      configuration['max_threshold'] = self.threshold.get()
    except tk.TclError:
      configuration['max_threshold'] = 0.1

  def update_detector(self, *kwargs):
    configuration['detector'] = self.selected_model.get()
    self.initialize_face_detector()

  def initialize_face_detector(self):
    detector_model = configuration['detector']
    if detector_model == "haar cascade":
      self.face_detector = HaarCascadeModel()
    elif detector_model == "mtcnn":
      self.face_detector = MTCNNModel()
    elif detector_model == "hog":
      self.face_detector = HOGModel()
    
  def show_webcam(self):
    _, frame = self.cap.read()
    bgr_image = frame = cv2.flip(frame, 1)

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = self.face_detector.detectMultipleFaces(bgr_image)
    if len(face_results) > 0:
      face_result = face_results[0]
      (x, y, w, h) = face_result
      start_point = (x, y)
      end_point = (x + w, y + h)
      rgb_image = cv2.rectangle(rgb_image, start_point, end_point, (255, 0, 0), 2)

      
      detected_image = bgr_image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
      dataset_path = "DATASET/"
      dfs = DeepFace.find(detected_image, dataset_path, enforce_detection=False)

      try:
        if (len(dfs) > 0):
          df = dfs[0].iloc[0]
          similar_image_path = df['identity']
          cosine_similarity = df['VGG-Face_cosine']
          label = os.path.basename(os.path.dirname(similar_image_path))
          font = cv2.FONT_HERSHEY_SIMPLEX
          font_scale = 1
          color = (204, 0, 0)
          thickness = 0
          origin_identitas = (50, 50)
          print(f"cosine similarity : {cosine_similarity}")
          if (cosine_similarity > configuration['max_threshold']):
            cv2.putText(rgb_image, f'wajah tidak dikenali', origin_identitas, font, font_scale, color, thickness)
          else:
            cv2.putText(rgb_image, f'identitas : {label}', origin_identitas, font, font_scale, color, thickness)
      except IndexError:
        pass

    # Mengkonversi gambar ke PhotoImage
    rgba_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2RGBA)
    rgb_array = rgb_array = Image.fromarray(rgba_image)
    photo_image = ImageTk.PhotoImage(image = rgb_array)
    self.webcam_label.photo_image = photo_image
    self.webcam_label.configure(image=photo_image)
    self.after_id = self.webcam_label.after(10, self.show_webcam)
    
  def process_destroy(self):
    self.webcam_label.after_cancel(self.after_id)
    self.destroy()
    self.cap.release()
    cv2.destroyAllWindows()
