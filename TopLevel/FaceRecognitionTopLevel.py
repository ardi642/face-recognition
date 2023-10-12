import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import os
from FaceDetectionModel.MTCNNModel import MTCNNModel
from FaceDetectionModel.HaarCascadeModel import HaarCascadeModel
from FaceDetectionModel.HOGModel import HOGModel

from HumanDetector import HumanDetector

from deepface import DeepFace
import logging, pickle, numpy as np

from configuration import configuration

import logging

root_logger = logging.getLogger()
root_logger.propagate = False  # Disable propagation

detection_model_options = ['haar cascade', 'mtcnn', 'hog']

def get_face_embedding(bgr_image):
  return DeepFace.represent(bgr_image, enforce_detection=False)[0]['embedding']

class FaceRecognitionTopLevel(tk.Toplevel):
  def __init__(self, master, **kwargs):
    super().__init__(master, **kwargs)
    self.geometry("1000x700")
    
    self.title("uji pengenalan wajah")
    self.cap = cv2.VideoCapture(0)
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    self.left_frame = tk.Frame(self)
    self.left_frame.pack(fill="both", expand=True, side="left")
    
    # load face recognition model
    with open('SVM_Model/svm_model.pkl', 'rb') as model_file:
      self.model = pickle.load(model_file)

    # self.humanDetector = HumanDetector()
    self.webcam_label = tk.Label(self.left_frame)
    self.webcam_label.pack(fill="both", expand=True, side="top")

    self.setting_frame = tk.Frame(self.left_frame)
    self.setting_frame.pack(fill="y", side="top", pady=(0, 25))

    self.selected_model = tk.StringVar()
    self.selected_model.trace("w", self.update_detector)

    self.detection_model_label = tk.Label(self.setting_frame, text='model pendeteksi wajah')
    self.detection_model_label.grid(column=0, row=1, padx=5, pady=5)
    self.detection_model_options = ttk.OptionMenu(self.setting_frame, self.selected_model,
                                                  detection_model_options[0], *detection_model_options)
    self.detection_model_options.grid(column=1, row=1, padx=5, pady=5)

    self.threshold_label = tk.Label(self.setting_frame, text='threshold maksimal cosine similarity')
    self.threshold_label.grid(column=0, row=2, padx=5, pady=5)

    self.isVerified = False
    self.after_id = None

    self.threshold = tk.DoubleVar()
    configuration['max_threshold'] = 0.1
    self.threshold.set(0.1)

    self.threshold_entry = tk.Entry(self.setting_frame, textvariable=self.threshold)
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

  def box_detected_image(self, rgb_image, bounding_box):
    (x, y, w, h) = bounding_box
    start_point = (x, y)
    end_point = (x + w, y + h)
    rgb_image = cv2.rectangle(rgb_image, start_point, end_point, (255, 0, 0), 2)
  
  def get_detected_image(self, bgr_image, bounding_box):
    (x, y, w, h) = bounding_box
    start_point = (x, y)
    end_point = (x + w, y + h)
    detected_image = bgr_image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
    return detected_image
  
  def identify_detected_image(self, rgb_image, detected_image):
    image_embedding = get_face_embedding(detected_image)
    arr_image_embedding = [image_embedding]
    label = self.model.predict(arr_image_embedding)[0]
    probabilities = self.model.predict_proba(arr_image_embedding)[0]
    label_probability = f"{(np.max(probabilities) * 100):.3f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (204, 0, 0)
    thickness = 0
    cv2.putText(rgb_image, f'identitas : {label}', (50, 50), font, font_scale, color, thickness)
    cv2.putText(rgb_image, f'probability : {label_probability}%', (25, 25), font, font_scale, color, thickness)

  def set_image_in_label(self, rgb_image):
      # Mengkonversi gambar ke PhotoImage``
      rgba_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2RGBA)
      rgb_array = rgb_array = Image.fromarray(rgba_image)
      photo_image = ImageTk.PhotoImage(image = rgb_array)
      self.webcam_label.photo_image = photo_image
      self.webcam_label.configure(image=photo_image)
    
  def show_webcam(self):
    _, frame = self.cap.read()
    bgr_image = frame = cv2.flip(frame, 1)
    modified_rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    face_results = self.face_detector.detectMultipleFaces(bgr_image)
    # jika wajah terdeteksi

    if len(face_results) > 0:
      bounding_box = face_results[0]
      # method ini memodifikasi isi variabel modified_rgb_image
      detected_image = self.get_detected_image(bgr_image, bounding_box)
      self.box_detected_image(modified_rgb_image, bounding_box)
      # method ini memodifikasi isi variabel modified_rgb_image
      self.identify_detected_image(modified_rgb_image, detected_image)      

      self.set_image_in_label(modified_rgb_image)
      self.after_id = self.after(10, self.show_webcam)

  def process_destroy(self):
    self.after_cancel(self.after_id)
    self.destroy()
    self.cap.release()
    cv2.destroyAllWindows()
