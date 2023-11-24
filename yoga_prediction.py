
import pandas
import csv
import cv2
import itertools
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import math

class_names=['chair','cobra','dog','tree','warrior']

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection 

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

pose_sample_rpi_path = os.path.join(os.getcwd(), 'examples/lite/examples/pose_estimation/raspberry_pi')
sys.path.append(pose_sample_rpi_path)

import cv2
import time

# Open the default camera
cap = cv2.VideoCapture(0)

# Check if camera is opened successfully
if not cap.isOpened():
    print("Unable to open the camera")
    exit()

# Wait for 10 seconds before capturing the image
wait_time = 10
start_time = time.time()
while time.time() - start_time < wait_time:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow('Video Feed', frame)

    # Wait for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Capture the image
ret, frame = cap.read()

# Release the camera
cap.release()

# Save the image
cv2.imwrite('captured_image1.jpg', frame)

# Display the captured image
cv2.imshow('Captured Image', frame)
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()


# Load MoveNet Thunder model
import utils
from data import BodyPart
from ml import Movenet
movenet = Movenet('movenet_thunder')

# Define function to run pose estimation using MoveNet Thunder.
# You'll apply MoveNet's cropping algorithm and run inference multiple times on
# the input image to improve pose estimation accuracy.
def detect(input_tensor, inference_count=3):
  """Runs detection on an input image.

  Args:
    input_tensor: A [height, width, 3] Tensor of type tf.float32.
      Note that height and width can be anything since the image will be
      immediately resized according to the needs of the model within this
      function.
    inference_count: Number of times the model should run repeatly on the
      same input image to improve detection accuracy.

  Returns:
    A Person entity detected by the MoveNet.SinglePose.
  """
  image_height, image_width, channel = input_tensor.shape

  # Detect pose using the full input image
  movenet.detect(input_tensor.numpy(), reset_crop_region=True)

  # Repeatedly using previous detection result to identify the region of
  # interest and only croping that region to improve detection accuracy
  for _ in range(inference_count - 1):
    person = movenet.detect(input_tensor.numpy(), 
                            reset_crop_region=False)

  return person

def draw_prediction_on_image(
    image, person, crop_region=None, close_figure=True,
    keep_input_size=False):

  # Draw the detection result on top of the image.
  image_np = utils.visualize(image, [person])

  # Plot the image with detection results.
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  im = ax.imshow(image_np)

  if close_figure:
    plt.close(fig)

  if not keep_input_size:
    image_np = utils.keep_aspect_ratio_resizer(image_np, (512, 512))

  return image_np


is_skip_step_1 = False #@param ["False", "True"] {type:"raw"}


class MoveNetPreprocessor1(object):
    """Helper class to preprocess pose sample images for classification."""

    def __init__(self, images_out_folder, csv_out_path):
        """Creates a preprocessor to detect pose from a single image and save the result as a CSV.

        Args:
            images_out_folder: Path to write the image overlay with detected landmarks. This image is useful when you need to debug accuracy issues.
            csv_out_path: Path to write the CSV containing the detected landmark coordinates and label of the image that can be used to train a pose classification model.
        """
        self._images_out_folder = images_out_folder
        self._csv_out_path = csv_out_path
        self._messages = []

    def process(self, image_path, detection_threshold=0.1):
        """Preprocesses a single image.

        Args:
            image_path: Path to the input image.
            detection_threshold: Only keep the image if all landmark confidence scores are above this threshold.
        """
        # Detect landmarks in the image and write it to a CSV file
        with open(self._csv_out_path, 'w') as csv_out_file:
            csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

            try:
                image = tf.io.read_file(image_path)
                image = tf.io.decode_jpeg(image)
            except:
                self._messages.append('Skipped ' + image_path + '. Invalid image.')
            else:
                image_height, image_width, channel = image.shape

                # Skip images that aren't RGB because Movenet requires RGB images
                if channel != 3:
                    self._messages.append('Skipped ' + image_path + '. Image isn\'t in RGB format.')
                else:
                    person = detect(image)

                    # Save landmarks if all landmarks were detected
                    min_landmark_score = min([keypoint.score for keypoint in person.keypoints])
                    should_keep_image = min_landmark_score >= detection_threshold
                    if not should_keep_image:
                        self._messages.append('Skipped ' + image_path + '. No pose was confidently detected.')
                    else:
                        # Draw the prediction result on top of the image for debugging later
                        output_overlay = draw_prediction_on_image(
                            image.numpy().astype(np.uint8), person, close_figure=True, keep_input_size=True)

                        # Write detection result into an image file
                        output_frame = cv2.cvtColor(output_overlay, cv2.COLOR_RGB2BGR)
                        filename = os.path.basename(image_path)
                        cv2.imwrite(os.path.join(self._images_out_folder, filename), output_frame)

                        # Get landmarks and scale them to the same size as the input image
                        pose_landmarks = np.array(
                            [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score] for keypoint in person.keypoints],
                            dtype=np.float32)

                        # Write the landmark coordinates to the CSV file
                        coordinates = pose_landmarks.flatten().astype(str).tolist()
                        csv_out_writer.writerow([os.path.basename(image_path)] + coordinates)

        if not os.path.exists(self._images_out_folder):
            os.makedirs(self._images_out_folder)

        # Print the error message collected
        for message in self._messages:
            print(message, file=sys.stderr)

if not is_skip_step_1:
  images_out_train_folder = 'abc'
  csvs_out_train_path = 'train_data4.csv'

  preprocessor = MoveNetPreprocessor1(
      
      images_out_folder=images_out_train_folder,
      csv_out_path=csvs_out_train_path,
  )

  preprocessor.process('captured_image1.jpg')


from tensorflow.keras.models import load_model

# load the model
model = load_model('mymodelmain.h5')


csv_path='train_data4.csv'
df = pd.read_csv(csv_path, header=None)

# Extract landmark coordinates as numpy array
landmarks = df.iloc[:, 1:].values.astype(np.float32)

# Normalize landmark coordinates by subtracting mean and dividing by standard deviation
mean = landmarks.mean(axis=0)
std = landmarks.std(axis=0)

y_pred1 = model.predict(landmarks)

# Convert the prediction result to class name
y_pred_label = [class_names[i] for i in np.argmax(y_pred1, axis=1)]
print(y_pred_label)
      

def calculateAngle(landmark1, landmark2, landmark3, landmark4, landmark5, landmark6):

    #landmark coordinates 
    x1 = landmark1
    y1 = landmark2
    x2 = landmark3
    y2 = landmark4
    x3 = landmark5
    y3 = landmark6

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle < 0
    if angle < 0:

        # Add 360 to the angle
        angle += 360
    
    # Return the calculated angle
    return angle     


import pandas as pd

df = pd.read_csv(csv_path, header=None)
 
left_elbow_angle = calculateAngle(df[16],df[17],df[22],df[23],df[28],df[29])
                                   
# Get the angle between the right shoulder, elbow and wrist points. 
right_elbow_angle = calculateAngle(df[19],df[20],df[25],df[26],df[31],df[32]) 

# Get the angle between the left elbow, shoulder and hip points. 
left_shoulder_angle = calculateAngle(df[22],df[23],df[16],df[17],df[34],df[35])

# Get the angle between the right hip, shoulder and elbow points. 
right_shoulder_angle = calculateAngle(df[37],df[38],df[19],df[20],df[25],df[26])

# Get the angle between the left hip, knee and ankle points. 
left_knee_angle = calculateAngle(df[34],df[35],df[40],df[41],df[46],df[47])
# Get the angle between the right hip, knee and ankle points 
right_knee_angle = calculateAngle(df[37],df[38],df[43],df[44],df[49],df[50])

print(left_elbow_angle)
print(right_elbow_angle)
print(left_knee_angle)
print(right_knee_angle)

if y_pred_label == ['tree'] :
   if right_knee_angle > 30 and right_knee_angle < 80 :
      
      if  left_elbow_angle > 305 and left_elbow_angle < 335 and right_elbow_angle > 20 and right_elbow_angle < 60:
         
         print('The pose is accurate')

      else:
         
         print('Bend the arms properly')
        
   else:
      
      if  left_elbow_angle > 305 and left_elbow_angle < 335 and right_elbow_angle > 20 and right_elbow_angle < 70:
         
         print('The leg is not bent at the right angle')

      else:
         
         print('The leg is not bent at the correct angle and arms are not bent correctly')
      
elif y_pred_label == ['chair'] :
   if left_knee_angle > 240 and left_knee_angle < 300 and right_knee_angle > 55 and right_knee_angle < 125 :
      
      if  left_elbow_angle > 160 and left_elbow_angle < 200 and right_elbow_angle > 160 and right_elbow_angle < 200:
         
         print('The pose is accurate')

      else:
         
         print('Keep your arms straight ')
        
   else:
      
      if  left_elbow_angle > 160 and left_elbow_angle < 200 and right_elbow_angle > 160 and right_elbow_angle < 200:
         
         print('Bend your legs at right angle ! ')

      else:
         
         print('Straighten your arms and bend your knees at the right angle!')

 

