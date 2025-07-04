import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Paths
FRAMES_DIR = 'extracted_frames'
KEYPOINTS_DIR = 'extracted_keypoints'  # Directory for keypoints

# MediaPipe setup
mp_holistic = mp.solutions.holistic

# Helper to extract keypoints from a frame
def extract_keypoints(results):
    # Pose
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # Left hand
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    # Right hand
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

print("Current working directory:", os.getcwd())
print("Contents of extracted_frames:", os.listdir(FRAMES_DIR))

if not os.path.exists(KEYPOINTS_DIR):
    os.makedirs(KEYPOINTS_DIR)

print("Starting MediaPipe holistic processing...")

labels = os.listdir(FRAMES_DIR)
print(f"Number of labels found: {len(labels)}")
#print(f"First 5 labels: {labels[:5]}")

with mp_holistic.Holistic(static_image_mode=True) as holistic:
    for label in labels:
        print(f"About to process label: {label}")
        label_path = os.path.join(FRAMES_DIR, label)
        if not os.path.isdir(label_path):
            continue
        frames = sorted([f for f in os.listdir(label_path) if f.lower().endswith('.jpg')])
        #print(f"First 5 frames in {label}: {frames[:5]}")
        #print(f"Total frames in {label}: {len(frames)}")
        keypoints_label_dir = os.path.join(KEYPOINTS_DIR, label)
        try:
            os.makedirs(keypoints_label_dir, exist_ok=True)
        except Exception as e:
            print(f"Failed to create directory {keypoints_label_dir}: {e}")
            continue
        keypoints_saved = 0
        frames_deleted = 0
        try:
            for idx, frame_name in enumerate(frames):
                frame_path = os.path.join(label_path, frame_name)
                try:
                    image = cv2.imread(frame_path)
                    if image is None:
                        print(f"Warning: Could not read {frame_path}")
                        continue
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image_rgb)
                    if results.pose_landmarks is not None:
                        keypoints = extract_keypoints(results)
                        npy_name = os.path.splitext(frame_name)[0] + '.npy'
                        save_path = os.path.join(keypoints_label_dir, npy_name)
                        np.save(save_path, keypoints)
                        print(f"Saved keypoints (in try): {save_path}")
                        keypoints_saved += 1
                    else:
                        os.remove(frame_path)
                        frames_deleted += 1
                except Exception as e:
                    print(f"Exception for {frame_path}: {e}")
                    # Try to save dummy keypoints if possible
                    try:
                        if 'keypoints' in locals():
                            npy_name = os.path.splitext(frame_name)[0] + '.npy'
                            save_path = os.path.join(keypoints_label_dir, npy_name)
                            np.save(save_path, keypoints)
                            print(f"Saved keypoints (in except): {save_path}")
                            keypoints_saved += 1
                    except Exception as save_e:
                        print(f"Failed to save keypoints in except for {frame_path}: {save_e}")
                if idx % 100 == 0:
                    print(f"Processed {idx} frames in {label}...")
        except Exception as e:
            print(f"Error during frame processing for label {label}: {e}")
        print(f"Label '{label}': {keypoints_saved} keypoints saved, {frames_deleted} frames deleted (no human figure found)")

#print("Contents of extracted_frames:", os.listdir(FRAMES_DIR))
