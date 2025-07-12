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
import pickle
from collections import deque

# Set the path to the data directory
PATH = os.path.join('data')

# Load the trained model
model = load_model('isl_sign_language_model.h5')
print("Model loaded successfully")

# Load the label map from pickle file
LABEL_MAP_PATH = 'isl_label_map.pkl'
if os.path.exists(LABEL_MAP_PATH):
    with open(LABEL_MAP_PATH, 'rb') as f:
        label_map = pickle.load(f)
    # Create reverse mapping from index to label
    actions = np.array([label for label, idx in sorted(label_map.items(), key=lambda x: x[1])])
    print(f"Loaded gesture translations: {actions}")
    print(f"Total number of gestures: {len(actions)}")
else:
    print("Warning: isl_label_map.pkl not found!")
    actions = np.array([])

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

SEQ_LEN = 30  # or whatever you used for training
keypoints_buffer = []
last_pred_idx = None
prediction_text = "Loading..."

# Improved prediction handling
prediction_buffer = deque(maxlen=5)  # Store last 5 predictions for smoothing
confidence_threshold = 0.5  # Lowered from 0.7 for better sensitivity
prediction_stability_threshold = 3  # How many consistent predictions needed

# Center positioning variables
center_prompt_active = False
center_prompt_timer = 0
CENTER_PROMPT_DURATION = 30  # Show prompt for 30 frames

# Colors for grid lines (BGR)
GRID_COLOR_DEFAULT = (192, 192, 192)  # Light Grey
GRID_COLOR_HIGHLIGHT = (0, 255, 0)   # Green

def draw_hindi_text(img, text, position, font_size=60, color=(0,255,0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    try:
        font = ImageFont.truetype("C:\\Windows\\Fonts\\Nirmala.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def get_most_common_prediction(prediction_buffer):
    """Get the most common prediction from the buffer for stability"""
    if len(prediction_buffer) < prediction_stability_threshold:
        return None, 0
    
    # Count occurrences of each prediction
    pred_counts = {}
    for pred in prediction_buffer:
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
    
    # Find the most common prediction
    most_common = max(pred_counts.items(), key=lambda x: x[1])
    
    # Only return if it appears at least stability_threshold times
    if most_common[1] >= prediction_stability_threshold:
        return most_common[0], most_common[1] / len(prediction_buffer)
    
    return None, 0

def check_center_positioning(results, frame_width, frame_height):
    """Check if person is properly centered in the frame using middle grid lines"""
    if not results.pose_landmarks:
        return False
    
    # Get nose position (landmark 0)
    nose = results.pose_landmarks.landmark[0]
    nose_x = nose.x * frame_width
    
    # Define center region using middle two grid lines (1/3 and 2/3 of frame width)
    center_x_min = frame_width * (1/3)  # First vertical grid line
    center_x_max = frame_width * (2/3)  # Second vertical grid line
    
    # Check if nose is between the middle two grid lines
    in_center = (center_x_min <= nose_x <= center_x_max)
    
    return in_center

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints(results):
    # Pose
    pose = np.zeros(33 * 4)
    try:
        if results.pose_landmarks and len(results.pose_landmarks.landmark) > 0:
            pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
    except Exception as e:
        pass
    
    # Face
    face = np.zeros(468 * 3)
    try:
        if results.face_landmarks and len(results.face_landmarks.landmark) > 0:
            face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten()
    except Exception as e:
        pass
    
    # Left hand
    lh = np.zeros(21 * 3)
    try:
        if results.left_hand_landmarks and len(results.left_hand_landmarks.landmark) > 0:
            lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
    except Exception as e:
        pass
    
    # Right hand
    rh = np.zeros(21 * 3)
    try:
        if results.right_hand_landmarks and len(results.right_hand_landmarks.landmark) > 0:
            rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
    except Exception as e:
        pass
    
    result = np.concatenate([pose, face, lh, rh])
    return result

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame from camera.")
            time.sleep(0.05)
            continue

        # Flip the frame for a mirror effect
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w, _ = image.shape

        # Check center positioning
        is_centered = check_center_positioning(results, w, h)
        
        if not is_centered:
            center_prompt_active = True
            center_prompt_timer = CENTER_PROMPT_DURATION
        else:
            if center_prompt_timer > 0:
                center_prompt_timer -= 1
            if center_prompt_timer == 0:
                center_prompt_active = False

        # Draw vertical center line
        center_x = w // 2
        cv2.line(image, (center_x, 0), (center_x, h), (200, 200, 200), 1)

        # Draw grid lines (3x3 grid)
        num_grid = 3
        for i in range(1, num_grid):
            x = w * i // num_grid
            cv2.line(image, (x, 0), (x, h), (192, 192, 192), 1)
            y = h * i // num_grid
            cv2.line(image, (0, y), (w, y), (192, 192, 192), 1)

        # Draw face landmarks (original coloring)
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
            )

        # Draw right hand landmarks
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
            )

        # Draw left hand landmarks
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            )

        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        # Extract keypoints and predict gesture
        keypoints = extract_keypoints(results)
        keypoints_buffer.append(keypoints)
        if len(keypoints_buffer) > SEQ_LEN:
            keypoints_buffer.pop(0)

        if len(keypoints_buffer) == SEQ_LEN:
            input_data = np.expand_dims(keypoints_buffer, axis=0).astype(np.float32)
            prediction = model.predict(input_data, verbose=0)
            pred_idx = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Add current prediction to buffer
            prediction_buffer.append(pred_idx)
            
            # Get stable prediction from buffer
            stable_pred_idx, stability_score = get_most_common_prediction(prediction_buffer)
            
            if confidence > confidence_threshold and stable_pred_idx is not None:
                if stable_pred_idx < len(actions):
                    if last_pred_idx != stable_pred_idx:
                        print(f"Predicted gesture: {actions[stable_pred_idx]} (Confidence: {confidence*100:.1f}%, Stability: {stability_score*100:.1f}%)")
                        last_pred_idx = stable_pred_idx
                    
                    # Color code based on confidence and stability
                    if confidence > 0.8 and stability_score > 0.8:
                        color = (0, 255, 0)  # Green for high confidence and stability
                    elif confidence > 0.6 and stability_score > 0.6:
                        color = (0, 255, 255)  # Yellow for moderate confidence
                    else:
                        color = (0, 165, 255)  # Orange for lower confidence
                    
                    prediction_text = f"{actions[stable_pred_idx]} ({confidence*100:.1f}%)"
                else:
                    print(f"Warning: Prediction index {stable_pred_idx} is out of bounds for actions array of size {len(actions)}")
                    prediction_text = "Unknown Gesture"
                    color = (0, 0, 255)  # Red for unknown
            else:
                if len(prediction_buffer) < prediction_stability_threshold:
                    prediction_text = "Building up predictions..."
                else:
                    prediction_text = "Detecting..."
                color = (128, 128, 128)  # Gray for detecting
        else:
            prediction_text = f"Loading... ({len(keypoints_buffer)}/{SEQ_LEN})"
            color = (128, 128, 128)

        # Display prediction with color coding
        cv2.putText(image, prediction_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Display additional info
        info_text = f"Buffer: {len(prediction_buffer)}/5 | Threshold: {confidence_threshold}"
        cv2.putText(image, info_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Display center positioning prompt
        if center_prompt_active:
            center_text = "Please stand in the center of the frame"
            cv2.putText(image, center_text, (w//2 - 200, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Sign Language Live', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Pressed 'q', exiting...")
            break
except KeyboardInterrupt:
    print("Keyboard interrupt received. Exiting...")
except Exception as e:
    print("Exception occurred:", e)
    import traceback
    traceback.print_exc()  # This will show the exact line where the error occurs
finally:
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
