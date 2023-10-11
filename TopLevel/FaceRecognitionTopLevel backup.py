import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import keras
import os
from keras.models import load_model
from FaceDetectionModel.MTCNNModel import MTCNNModel
from FaceDetectionModel.HaarCascadeModel import HaarCascadeModel
from FaceDetectionModel.HOGModel import HOGModel

from deepface import DeepFace

from configuration import configuration

detection_model_options = ['haar cascade', 'mtcnn', 'hog']

class FaceRecognitionTopLevel(tk.Toplevel):
  def __init__(self, master, **kwargs):
    super().__init__(master, **kwargs)
    self.geometry("1000x700")
    
    self.title("uji pengenalan wajah")
    self.cap = cv2.VideoCapture(0)
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    self.img_height = 128
    self.img_width = 128
    self.model = load_model('model_pengenalan_wajah.h5')
    self.daftar_kelas = os.listdir('DATASET/WAJAH/TRAINING/')
    self.daftar_kelas.sort()
    
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

    self.show_webcam()

    self.protocol("WM_DELETE_WINDOW", self.process_destroy)

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
    self.gambar = gbr_img = cv2.flip(frame, 1)

    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_img = rgb_img.copy()

    self.initialize_face_detector()

    face_results = self.face_detector.detectMultipleFaces(self.gambar)
    # gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # face_cascade = cv2.CascadeClassifier("HAAR CASCADE DATASET/haarcascade_frontalface_default.xml")
    # data_wajah = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=9)
    if len(face_results) > 0:
      face_result = face_results[0]
      (x, y, w, h) = face_result
      self.start_point = (x, y)
      self.end_point = (x + w, y + h)
      result_img = cv2.rectangle(result_img, self.start_point, self.end_point, (255, 0, 0), 2)

      dfs = DeepFace.find(self.get_gambar_terdeteksi(), "DATASET/WAJAH/TRAINING")
      print(dfs)
      

      # try:
      #   one_hot_encoding_prediksi = self.model.predict(self.get_gambar_terdeteksi())[0]
      #   index_kelas_prediksi = np.argmax(one_hot_encoding_prediksi)
      #   probabilitas_prediksi = one_hot_encoding_prediksi[index_kelas_prediksi]
      #   nama_kelas_prediksi = self.daftar_kelas[index_kelas_prediksi]
        
      #   font = cv2.FONT_HERSHEY_SIMPLEX
      #   font_scale = 1
      #   color = (204, 0, 0)
      #   thickness = 0
      #   origin_identitas = (50, 50)
      #   origin_probabilitas = (50, 70)
      #   cv2.putText(result_img, f'identitas : {nama_kelas_prediksi}', origin_identitas, font, font_scale, color, thickness)
      #   cv2.putText(result_img, f'probabilitas : {probabilitas_prediksi}%', origin_probabilitas, font, font_scale, color, thickness)
      # except:
      #   pass

    # Mengkonversi gambar ke PhotoImage
    rgba_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2RGBA)
    rgb_array = rgb_array = Image.fromarray(rgba_img)
    photo_image = ImageTk.PhotoImage(image = rgb_array)
    self.webcam_label.photo_image = photo_image
    self.webcam_label.configure(image=photo_image)
    self.after_id = self.webcam_label.after(10, self.show_webcam)
    
  def get_gambar_terdeteksi(self):
    if self.start_point is None:
      return None
    
    start_point = self.start_point
    end_point = self.end_point
    gambar_deteksi = self.gambar[start_point[1]:end_point[1], start_point[0]:end_point[0]]
    gambar_deteksi = cv2.resize(gambar_deteksi,(self.img_height, self.img_width))
    # gambar_deteksi = cv2.cvtColor(gambar_deteksi, cv2.COLOR_BGR2RGB)
    array_gambar = [gambar_deteksi]
    array_gambar = np.array(array_gambar)
    array_gambar = array_gambar / 255
    return array_gambar

  def process_destroy(self):
    self.webcam_label.after_cancel(self.after_id)
    self.destroy()
    self.cap.release()
    cv2.destroyAllWindows()
