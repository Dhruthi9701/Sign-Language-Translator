import os
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from collections import Counter

class KeypointsSequence(Sequence):
    def __init__(self, keypoints_base, batch_size=8, max_seq_len=None):
        self.samples = []
        self.labels = []
        self.label_map = {}
        self.batch_size = batch_size

        # Collect all .npy files and their labels (translation names from subfolders)
        all_labels = set()
        file_label_pairs = []
        
        print(f"Loading data from: {keypoints_base}")
        
        for folder in os.listdir(keypoints_base):
            folder_path = os.path.join(keypoints_base, folder)
            if os.path.isdir(folder_path):
                print(f"Processing alphabet folder: {folder}")
                # Look for subfolders (translation names) inside each alphabet folder
                for subfolder in os.listdir(folder_path):
                    subfolder_path = os.path.join(folder_path, subfolder)
                    if os.path.isdir(subfolder_path):
                        npy_files = [f for f in os.listdir(subfolder_path) if f.endswith('.npy')]
                        print(f"  {subfolder}: {len(npy_files)} samples")
                        for file in npy_files:
                            all_labels.add(subfolder)
                            file_label_pairs.append((os.path.join(subfolder_path, file), subfolder))
        
        # Build label map from unique translation names
        sorted_labels = sorted(list(all_labels))
        self.label_map = {label: idx for idx, label in enumerate(sorted_labels)}
        
        # Assign labels to samples
        for file_path, label in file_label_pairs:
            self.samples.append(file_path)
            self.labels.append(self.label_map[label])

        # Shuffling
        combined = list(zip(self.samples, self.labels))
        np.random.shuffle(combined)
        self.samples, self.labels = zip(*combined)

        # max sequence length and feature dimension
        if max_seq_len is None:
            self.max_seq_len = max(np.load(f).shape[0] for f in self.samples[:100])  # check first 100 for speed
        else:
            self.max_seq_len = max_seq_len
        self.feature_dim = np.load(self.samples[0]).shape[1]

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, idx):
        batch_samples = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = []
        for file in batch_samples:
            arr = np.load(file)
            # Pad or truncate to max_seq_len
            if arr.shape[0] < self.max_seq_len:
                pad_width = ((0, self.max_seq_len - arr.shape[0]), (0, 0))
                arr = np.pad(arr, pad_width, mode='constant')
            else:
                arr = arr[:self.max_seq_len]
            X.append(arr.astype(np.float32))
        X = np.array(X)
        y = np.array(batch_labels)
        return X, y


keypoints_base = 'ISL_Keypoints'


# MODEL TRAINING


def main():
    print("Starting Sign Language Model Training...")
    print(f"Dataset path: {keypoints_base}")
    
    # Check if dataset exists
    if not os.path.exists(keypoints_base):
        print(f"Error: Dataset not found at {keypoints_base}")
        print("Please check the keypoints_base path")
        return
    
    batch_size = 8
    
    # Create generator
    print("\nLoading dataset...")
    train_gen = KeypointsSequence(keypoints_base, batch_size=batch_size)
    
    if len(train_gen.samples) == 0:
        print("No samples found! Please check your dataset structure.")
        return
    
    print(f"\nDataset loaded successfully!")
    print(f"Total samples: {len(train_gen.samples)}")
    print(f"Number of classes: {len(train_gen.label_map)}")
    print(f"Feature dimension: {train_gen.feature_dim}")
    print(f"Max sequence length: {train_gen.max_seq_len}")
    
    # Model
    input_shape = (train_gen.max_seq_len, train_gen.feature_dim)
    num_classes = len(train_gen.label_map)
    
    print(f"\nBuilding model...")
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    
    model = Sequential([
        Masking(mask_value=0., input_shape=input_shape),
        LSTM(64, return_sequences=False),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Print model summary
    model.summary()

    # Training
    print(f"\nStarting training...")
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    history = model.fit(train_gen, epochs=100, callbacks=[es])
    
    # Evaluate and print accuracy in percentage
    print(f"\nEvaluating model...")
    loss, acc = model.evaluate(train_gen)
    print(f"Final Accuracy: {acc*100:.2f}%")
    
    # Save model and label map
    model.save('isl_sign_language_model.h5')
    with open('isl_label_map.pkl', 'wb') as f:
        pickle.dump(train_gen.label_map, f)
    
    print(f"\nModel saved as: isl_sign_language_model.h5")
    print(f"Label map saved as: isl_label_map.pkl")
    
    # Print data summary for debugging
    print(f"\nFinal Summary:")
    print("=" * 50)
    print(f"Label map size: {len(train_gen.label_map)}")
    print(f"Number of samples: {len(train_gen.samples)}")
    print(f"Labels distribution: {Counter(train_gen.labels)}")
    
    # Show some sample labels
    print(f"\nSample gestures in dataset:")
    for i, (label, idx) in enumerate(list(train_gen.label_map.items())[:10]):
        print(f"  {idx}: {label}")
    if len(train_gen.label_map) > 10:
        print(f"  ... and {len(train_gen.label_map) - 10} more")
    
    arr = np.load(train_gen.samples[0])
    print(f"\nSample data shape: {arr.shape}")
    print(f"Sample data type: {arr.dtype}")

# Run the training
if __name__ == "__main__":
    main()
