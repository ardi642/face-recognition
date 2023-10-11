import tkinter as tk
from PIL import Image, ImageTk
import cv2
from FaceDetectionModel.HaarCascadeModel import HaarCascadeModel
from FaceDetectionModel.MTCNNModel import MTCNNModel
from FaceDetectionModel.HOGModel import HOGModel
import os
from tkinter import messagebox, ttk

from configuration import configuration

detection_model_options = ['haar cascade', 'mtcnn', 'hog']

class WebcamTopLevel(tk.Toplevel):
  def __init__(self, master, captured_images, dataset_status, **kwargs):
    super().__init__(master, **kwargs)
    self.geometry("1000x700")
    self.dataset_status = dataset_status
    
    self.title(f"webcam dataset {self.dataset_status}")
    self.cap = cv2.VideoCapture(0)
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    self.captured_images = captured_images
    
    self.is_save_enabled = False
    self.total_saved_images = 0

    self.image_count_key = f"{self.dataset_status}_image_count"

    self.count_info_label = tk.Label(self, text="")
    self.count_info_label.pack(fill="y", side="top", anchor="s", pady=(10, 20))
    

    self.webcam_label = tk.Label(self)
    self.webcam_label.pack(fill="both", side="top", pady=(0, 20))

    self.bottom_frame = tk.Frame(self)
    self.bottom_frame.pack(fill="y", side="top", pady=(0, 25))

    self.identitas_label = tk.Label(self.bottom_frame, text='identitas')
    self.identitas_label.grid(column=0, row=0, padx=5, pady=5)

    self.identitas_entry = tk.Entry(self.bottom_frame, width=30)
    self.identitas_entry.grid(column=1, row=0, pady=5, padx=5)

    self.selected_model = tk.StringVar()
    self.selected_model.trace("w", self.update_detector)

    self.detection_model_label = tk.Label(self.bottom_frame, text='model pendeteksi wajah')
    self.detection_model_label.grid(column=0, row=1, padx=5, pady=5)
    self.detection_model_options = ttk.OptionMenu(self.bottom_frame, self.selected_model,
                                                  detection_model_options[0], *detection_model_options)
    self.detection_model_options.grid(column=1, row=1, padx=5, pady=5)

    self.save_button = tk.Button(self.bottom_frame, text="simpan dataset", command=self.enable_save)
    self.save_button.grid(column=0, row=2, columnspan=2, pady=5, padx=5)
    
    self.initialize_face_detector()

    self.update_webcam_frame()

    self.protocol("WM_DELETE_WINDOW", self.process_destroy)

  def update_detector(self, *kwargs):
    configuration['detector'] = self.selected_model.get()
    self.initialize_face_detector()

  def enable_save(self):
    if self.identitas_entry.get() == "":
      messagebox.showwarning(
        title='Peringatan',
        message="identitas belum dimasukkan"
      )
      self.focus()
      return
    self.is_save_enabled = True
    self.total_saved_images = 0
    self.save_button.config(state="disabled")
    self.identitas_entry.config(state="disabled")

  def initialize_face_detector(self):
    detector_model = configuration['detector']
    if detector_model == "haar cascade":
      self.face_detector = HaarCascadeModel()
    elif detector_model == "mtcnn":
      self.face_detector = MTCNNModel()
    elif detector_model == "hog":
      self.face_detector = HOGModel()
    
    
  def update_webcam_frame(self):
    # image_size = (480, 640, 3)
    _, frame = self.cap.read()
    # frame_image = cv2.resize(frame, (image_size[1], image_size[0]))
    frame_image = cv2.flip(frame, 1)

    rgb_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
    result_image = rgb_image.copy()

    face_results = self.face_detector.detectMultipleFaces(rgb_image)

    if len(face_results) > 0:
      # face result atau bounding box
      face_result = face_results[0]
      try:
        additional_bottom = configuration['additional_bottom']
      except KeyError:
        additional_bottom = 0

      (x, y, w, h) = face_result
      self.start_point = (x, y)
      self.end_point = (x + w, y + h + additional_bottom)
      result_image = cv2.rectangle(result_image, self.start_point, self.end_point, (255, 0, 0), 2)

      if self.is_save_enabled:
        if self.total_saved_images == configuration[self.image_count_key]:
          self.is_save_enabled = False
          self.total_saved_images = 0
          self.count_info_label.config(text="")
          self.save_button.config(state="normal")
          self.identitas_entry.config(state="normal")
          messagebox.showinfo(
            title='berhasil',
            message=f"dataset {self.dataset_status} wajah berhasil disimpan"
          )
          self.focus()
        else:
          identity = self.identitas_entry.get()
          try:
            for captured_image in self.captured_images:
              captured_image.save(rgb_image, identity, face_result)
            self.total_saved_images += 1
            self.count_info_label.config(text=f"jumlah foto : {self.total_saved_images}")
          except cv2.error:
            pass


    # Mengkonversi gambar ke PhotoImage
    rgba_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2RGBA)
    rgb_array = rgb_array = Image.fromarray(rgba_image)
    photo_image = ImageTk.PhotoImage(image = rgb_array)

    self.webcam_label.photo_image = photo_image
    self.webcam_label.configure(image=photo_image)
    self.after_id = self.webcam_label.after(configuration['refresh_time'], self.update_webcam_frame)

  def process_destroy(self):
    self.webcam_label.after_cancel(self.after_id)
    self.destroy()
    self.cap.release()
    cv2.destroyAllWindows()
