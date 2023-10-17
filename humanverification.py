import random
import keras
from tensorflow.keras.preprocessing import image

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math

# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
output_face_blendshapes=True,
output_facial_transformation_matrixes=True,
num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

verification_steps = [
  {
    'pose': 'kiri',
    'detail': 'geser wajah ke kiri'
  },
  {
    'pose': 'kanan',
    'detail': 'geser wajah ke kanan'
  },
  {
    'pose': 'bawah',
    'detail': 'arahkan wajah ke bawah'
  },
  {
    'pose': 'depan',
    'detail': 'arahkan wajah ke depan'
  },
  {
    'pose': 'atas',
    'detail': 'arahkan wajah ke depan'
  },
]

img_h = 480
img_w = 640

class HumanVerification():
  def __init__(self):
    self.set_verification_steps()

  def get_verification_steps(self):
    return self.verification_steps
  
  def set_verification_steps(self):
    self.verification_index = 0
    self.verification_steps = []
    self.verification_steps.append(verification_steps[0])
    self.verification_steps.append(verification_steps[1])
    self.verification_steps.append(verification_steps[2])
    # for i in range(5):
    #   random_number = random.randint(1, len(verification_steps)) - 1
    #   self.verification_steps.append(verification_steps[random_number])

  def add_verification_index(self, added_value = 1):
    self.verification_index += added_value
    return self.verification_index
  
  def get_verification_index(self):
    return len(self.verification_index)
  
  def get_current_step(self):
    return self.verification_steps[self.verification_index]

  def is_verified(self):
    return True if self.verification_index == len(self.verification_steps) else False

  def verify(self, face_landmarks, pose):
    detected_step = self.get_detected_step(face_landmarks)
    return True if detected_step['pose'] == pose else False
  
  def get_face_landmarks(self, rgb_image):
    rgb_image = cv2.resize(rgb_image, (img_w, img_h))
    rgb_image.flags.writeable = False
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    try:
      detection_result = detector.detect(mp_image)
      rgb_image.flags.writeable = True
      landmarks = detection_result.face_landmarks[0]
    except:
      landmarks = []
    
    return landmarks
  
  def get_face_webcam_distance(self, face_landmarks):
    if (len(face_landmarks) == 0):
      return None
    
    point_left = face_landmarks[145]
    point_left.x = int(point_left.x * img_w)
    point_left.y = int(point_left.y * img_h)

    point_right = face_landmarks[374]
    point_right.x = int(point_right.x * img_w)
    point_right.y = int(point_right.y * img_h)

    focal_length = 1 * img_w
    W = 6.3
    x1 = point_left.x
    x2 = point_right.x

    y1 = point_left.y
    y2 = point_right.y
    w = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    d = (W * focal_length) / w
    return d
  
  def get_detected_step(self, face_landmarks):
    face_3d = []
    face_2d = []
    if (len(face_landmarks) > 0):
      for idx, lm in enumerate(face_landmarks):
        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
          # if idx == 1:
          #     nose_2d = (lm.x * img_w, lm.y * img_h)
          #     nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

          x, y = int(lm.x * img_w), int(lm.y * img_h)

          # Get the 2D Coordinates
          face_2d.append([x, y])

          # Get the 3D Coordinates
          face_3d.append([x, y, lm.z])  

      # Convert it to the NumPy array
      face_2d = np.array(face_2d, dtype=np.float64)

      # Convert it to the NumPy array
      face_3d = np.array(face_3d, dtype=np.float64)

      # The camera matrix
      focal_length = 1 * img_w

      cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                              [0, focal_length, img_w / 2],
                              [0, 0, 1]])

      # The distortion parameters
      dist_matrix = np.zeros((4, 1), dtype=np.float64)

      # Solve PnP
      success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

      # Get rotational matrix
      rmat, jac = cv2.Rodrigues(rot_vec)

      # Get angles
      angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
      # yaw, pitch, roll = cv2.decomposeProjectionMatrix(rmat)

      # Get the y rotation degree
      x = angles[0] * 360
      y = angles[1] * 360
      z = angles[2] * 360

      # See where the user's head tilting
      if y < -25:
        # wajah ke kiri
        step = verification_steps[0]
      elif y > 15:
        # wajah ke kanan
        step = verification_steps[1]
      elif x < -10:
        # wajah ke bawah
        step = verification_steps[2]
      elif x > 15:
        # wajah ke depan
        step = verification_steps[3]
      else:
        # wajah ke atas
        step = verification_steps[4]

      return step
    
    return None