import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

from FaceDetectionModel.MTCNNModel import MTCNNModel
from FaceDetectionModel.HaarCascadeModel import HaarCascadeModel
from FaceDetectionModel.HOGModel import HOGModel

from deepface import DeepFace

import pickle, numpy as np

from humanverification import HumanVerification
from configuration import configuration

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
    
    # load face recognition model
    with open('SVM_Model/svm_model.pkl', 'rb') as model_file:
      self.model = pickle.load(model_file)

    self.webcam_label = tk.Label(self)
    self.webcam_label.pack(fill="both", expand=True, side="top")
    self.humanVerification = HumanVerification()
    self.bottom_frame = tk.Frame(self)
    self.bottom_frame.pack(fill="y", side="top", pady=(0, 25))

    self.selected_model = tk.StringVar()
    self.selected_model.trace("w", self.update_detector)

    self.detection_model_label = tk.Label(self.bottom_frame, text='model pendeteksi wajah')
    self.detection_model_label.grid(column=0, row=1, padx=5, pady=5)
    self.detection_model_options = ttk.OptionMenu(self.bottom_frame, self.selected_model,
                                                  detection_model_options[0], *detection_model_options)
    self.detection_model_options.grid(column=1, row=1, padx=5, pady=5)
    # self.threshold_label = tk.Label(self.bottom_frame, text='threshold maksimal cosine similarity')
    # self.threshold_label.grid(column=0, row=2, padx=5, pady=5)

    # self.threshold = tk.DoubleVar()
    # configuration['max_threshold'] = 0.1
    # self.threshold.set(0.1)

    # self.threshold_entry = tk.Entry(self.bottom_frame, textvariable=self.threshold)
    # self.threshold_entry.grid(column=1, row=2, pady=5, padx=5)

    # self.threshold.trace("w", self.update_threshold)

    self.initialize_face_detector()
    self.show_webcam()

    self.protocol("WM_DELETE_WINDOW", self.process_destroy)

  # def update_threshold(self, *args):
  #   try:
  #     configuration['max_threshold'] = self.threshold.get()
  #   except tk.TclError:
  #     configuration['max_threshold'] = 0.1

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

  def predict(self, bgr_detected_image):
      image_embedding = get_face_embedding(bgr_detected_image)
      arr_image_embedding = [image_embedding]
      label = self.model.predict(arr_image_embedding)[0]
      probabilities = self.model.predict_proba(arr_image_embedding)[0]
      label_probability = np.max(probabilities) * 100
      return {
        'label': label,
        'probability': label_probability
      }
    
  def show_webcam(self):
    _, frame = self.cap.read()
    bgr_image = frame = cv2.flip(frame, 1)

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (204, 0, 0)
    thickness = 0
    is_verified = self.humanVerification.is_verified()
    if not is_verified:
      face_landmarks = self.humanVerification.get_face_landmarks(rgb_image)
      current_step = self.humanVerification.get_current_step()

      pose = current_step['pose']
      face_webcam_distance = self.humanVerification.get_face_webcam_distance(face_landmarks)

      cv2.putText(rgb_image, current_step['detail'], (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
      if face_webcam_distance is None:  
        text = "wajah tidak terdeteksi"
        cv2.putText(rgb_image, text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
      elif face_webcam_distance < 60:
        text = "wajah tidak boleh dekat dari webcam untuk mencegah kecurangan absensi"
        cv2.putText(rgb_image, text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
      elif (self.humanVerification.verify(face_landmarks, pose)):
        self.humanVerification.add_verification_index()
        
    elif is_verified:
      face_results = self.face_detector.detectMultipleFaces(bgr_image)
      try:
        bounding_box = face_results[0]
      except IndexError:
        bounding_box = None

      if bounding_box is not None:
        (x, y, w, h) = bounding_box
        start_point = (x, y)
        end_point = (x + w, y + h)
        rgb_image = cv2.rectangle(rgb_image, start_point, end_point, (255, 0, 0), 2)

        detected_image = bgr_image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
        prediksi = self.predict(detected_image)

        cv2.putText(rgb_image, f'identitas : {prediksi["label"]}', (20, 50), font, font_scale, color, thickness)
        cv2.putText(rgb_image, f'probability : {prediksi["probability"]:.3f}%', (25, 25), font, font_scale, color, thickness)

      elif bounding_box is None:
        cv2.putText(rgb_image, f'wajah tidak dikenali', (20, 50), font, font_scale, color, thickness)

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
