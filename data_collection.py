import os
import re
import pandas as pd
import cv2
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import io
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn import metrics
import argparse
import zipfile

# --- CONFIGURATION ---
CSV_PATH = 'ISL_Dictionary_words.csv'
LINK_COLUMN = 'Link'
DOWNLOAD_DIR = 'downloaded_videos'
FRAMES_DIR = 'extracted_frames'
FRAME_EVERY_N = 1  # Save every nth frame

# --- GOOGLE DRIVE API SETUP ---
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate_drive():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('drive', 'v3', credentials=creds)

def extract_folder_id(url):
    match = re.search(r'/folders/([a-zA-Z0-9_-]+)', url)
    return match.group(1) if match else None

def list_files_in_folder(service, folder_id):
    query = f"'{folder_id}' in parents and mimeType contains 'video/' and trashed = false"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    return results.get('files', [])

def download_file(service, file_id, file_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, file_name)
    if os.path.exists(out_path):
        print(f"Already downloaded: {out_path}")
        return out_path
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(out_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    print(f"Downloaded: {out_path}")
    return out_path

def extract_frames(video_path, out_dir, every_n=1):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(out_dir, video_name)
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % every_n == 0:
            frame_path = os.path.join(save_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    print(f"Extracted {saved_count} frames from {video_path}")

def extract_data(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def load_data(keypoints_dir):
    labels = sorted([d for d in os.listdir(keypoints_dir) if os.path.isdir(os.path.join(keypoints_dir, d))])
    X, Y = [], []
    for label_idx, label in enumerate(labels):
        label_dir = os.path.join(keypoints_dir, label)
        for fname in os.listdir(label_dir):
            if fname.endswith('.npy'):
                X.append(np.load(os.path.join(label_dir, fname)))
                Y.append(label_idx)
    X = np.array(X)
    Y = to_categorical(Y, num_classes=len(labels))
    return X, Y, labels

def main(args):
    # If running on Vertex AI, unzip data from GCS
    if args.data_zip and args.data_dir:
        extract_data(args.data_zip, args.data_dir)
        keypoints_dir = os.path.join(args.data_dir, 'extracted_keypoints')
    else:
        keypoints_dir = args.keypoints_dir

    print(f"Loading data from: {keypoints_dir}")
    X, Y, labels = load_data(keypoints_dir)
    print(f"Loaded {X.shape[0]} samples, {len(labels)} classes.")

    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42, stratify=Y
    )

    # Build model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(len(labels), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    model.fit(X_train, Y_train, epochs=30, batch_size=32, validation_data=(X_test, Y_test))

    # Evaluate
    preds = np.argmax(model.predict(X_test), axis=1)
    true = np.argmax(Y_test, axis=1)
    accuracy = metrics.accuracy_score(true, preds)
    print(f"Test accuracy: {accuracy*100:.2f}%")

    # Save model
    model.save(args.model_output)
    print(f"Model saved to {args.model_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--keypoints_dir', type=str, default='extracted_keypoints',
                        help='Path to extracted_keypoints directory (local run)')
    parser.add_argument('--data_zip', type=str, default=None,
                        help='Path to zipped data (for Vertex AI, e.g., /gcs/path/extracted_keypoints.zip)')
    parser.add_argument('--data_dir', type=str, default='/tmp/data',
                        help='Where to extract zipped data (for Vertex AI)')
    parser.add_argument('--model_output', type=str, default='my_model.keras',
                        help='Where to save the trained model')
    args = parser.parse_args()
    main(args)