import numpy as np
import os
import string
import mediapipe as mp
import cv2
from my_functions import *
import keyboard
from tensorflow.keras.models import load_model
import language_tool_python
from PIL import ImageFont, ImageDraw, Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn import metrics
import json
import time

# Set the path to the data directory
PATH = os.path.join('data')

# Load action labels from extracted_keypoints directory
KEYPOINTS_DIR = 'extracted_keypoints'
if os.path.exists(KEYPOINTS_DIR):
    actions = np.array(sorted([d for d in os.listdir(KEYPOINTS_DIR) if os.path.isdir(os.path.join(KEYPOINTS_DIR, d))]))
else:
    actions = np.array([])

# Load the trained model
model = load_model('my_model.keras')
print("Model loaded successfully")

# Create an instance of the grammar correction tool
tool = language_tool_python.LanguageToolPublicAPI('en-UK')

# Initialize the lists
sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []

print("Attempting to open camera...")
cap = cv2.VideoCapture(0)
print("Camera object created.")
if not cap.isOpened():
    print("Cannot access camera.")
    exit()
print("Camera opened successfully")

keypoints_buffer = []
last_pred_idx = None
prediction_text = "Loading..."

# Colors for grid lines (BGR)
GRID_COLOR_DEFAULT = (192, 192, 192)  # Light Grey
GRID_COLOR_HIGHLIGHT = (0, 255, 0)   # Green

SEQ_LEN = 30  # or whatever value you used for training

def draw_hindi_text(img, text, position, font_size=60, color=(0,255,0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    try:
        font = ImageFont.truetype("C:\\Windows\\Fonts\\Nirmala.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic, \
     mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
    try:
        while True:
            ret, image = cap.read()
            if not ret:
                print('Warning: Failed to read frame from camera.')
                time.sleep(0.05)
                continue
            image = cv2.resize(image, (640, 480))

            # --- Background Blur using SelfieSegmentation ---
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            seg_results = selfie_segmentation.process(rgb_image)
            mask = seg_results.segmentation_mask > 0.5
            mask_3c = np.stack([mask]*3, axis=-1)  # Make mask 3-channel
            blurred = cv2.GaussianBlur(image, (55, 55), 0)
            image = np.where(mask_3c, image, blurred)
            # --- End Background Blur ---

            results, image = image_process(image, holistic)
            draw_landmarks(image, results)
            is_person_centered = False
            grid_color = GRID_COLOR_DEFAULT
            grid_thickness = 1
            if results.pose_landmarks:
                nose = results.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.NOSE]
                if (1/3 < nose.x < 2/3) and (1/3 < nose.y < 2/3):
                    is_person_centered = True
                    grid_color = GRID_COLOR_HIGHLIGHT
                    grid_thickness = 2
            height, width, _ = image.shape
            cv2.line(image, (width // 3, 0), (width // 3, height), grid_color, grid_thickness)
            cv2.line(image, (2 * width // 3, 0), (2 * width // 3, height), grid_color, grid_thickness)
            cv2.line(image, (0, height // 3), (width, height // 3), grid_color, grid_thickness)
            cv2.line(image, (0, 2 * height // 3), (width, 2 * height // 3), grid_color, grid_thickness)
            if is_person_centered:
                keypoints = keypoint_extraction(results)
                keypoints_buffer.append(keypoints)
                if len(keypoints_buffer) > SEQ_LEN:
                    keypoints_buffer.pop(0)
                if len(keypoints_buffer) == SEQ_LEN:
                    input_data = np.expand_dims(keypoints_buffer, axis=0).astype(np.float32)
                    prediction = model.predict(input_data, verbose=0)
                    pred_idx = np.argmax(prediction)
                    confidence = np.max(prediction)
                    if confidence > 0.7:
                        if last_pred_idx != pred_idx:
                            print(f"Predicted gesture: {actions[pred_idx]} ({confidence*100:.1f}%)")
                            last_pred_idx = pred_idx
                        prediction_text = f"{actions[pred_idx]} ({confidence*100:.1f}%)"
                    else:
                        prediction_text = "Detecting..."
                else:
                    prediction_text = "Loading..."
                cv2.putText(image, prediction_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                keypoints_buffer.clear()
                prediction_text = "Please stand in the center"
                cv2.putText(image, prediction_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Camera', image)
            cv2.waitKey(1)
    except Exception as e:
        print("Exception occurred:", e)
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tool.close()
