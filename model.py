import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add
from sklearn.utils import class_weight
from sklearn import metrics
import tensorflow as tf

# Path to your extracted keypoints
KEYPOINTS_DIR = 'extracted_keypoints'
SEQ_LEN = 30  # Number of frames per sequence

# Gather sequential data and labels
X_seq, y_seq = [], []
labels = sorted(os.listdir(KEYPOINTS_DIR))
for label in labels:
    label_dir = os.path.join(KEYPOINTS_DIR, label)
    if not os.path.isdir(label_dir):
        continue
    frame_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.npy')],
                         key=lambda x: int(''.join(filter(str.isdigit, x))))
    for i in range(len(frame_files) - SEQ_LEN + 1):
        seq = []
        valid = True
        for j in range(SEQ_LEN):
            arr = np.load(os.path.join(label_dir, frame_files[i + j]))
            if arr.shape != (258,):
                print(f"Warning: Skipping {frame_files[i + j]} in {label_dir} due to shape {arr.shape}")
                valid = False
                break
            seq.append(arr.astype(np.float32))
        if valid:
            X_seq.append(np.stack(seq))
            y_seq.append(label)

X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_seq)
y_cat = to_categorical(y_encoded)

# Normalize per feature (column-wise)
X_min = X_seq.min(axis=(0, 1), keepdims=True)
X_max = X_seq.max(axis=(0, 1), keepdims=True)
X_seq = (X_seq - X_min) / (X_max - X_min + 1e-8)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_cat, test_size=0.1, random_state=42, stratify=y_cat)

# Compute class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weights_dict = dict(enumerate(class_weights))

# Positional Encoding Layer
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, d_model):
        super().__init__()
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pos_encoding': self.pos_encoding.numpy()
        })
        return config

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

# Build a transformer-based model
input_shape = (SEQ_LEN, X_seq.shape[2])
inputs = Input(shape=input_shape)
x = PositionalEncoding(SEQ_LEN, X_seq.shape[2])(inputs)
# Transformer Encoder Block
attn_output = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
attn_output = Dropout(0.3)(attn_output)
attn_output = Add()([x, attn_output])
x = LayerNormalization(epsilon=1e-6)(attn_output)
# Feed Forward
ffn = Dense(128, activation='relu')(x)
ffn = Dropout(0.3)(ffn)
ffn = Dense(X_seq.shape[2], activation='relu')(ffn)
ffn_output = Add()([x, ffn])
x = LayerNormalization(epsilon=1e-6)(ffn_output)
# Pooling and classification
x = GlobalAveragePooling1D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(y_cat.shape[1], activation='softmax')(x)
model = Model(inputs, outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train without early stopping
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=128,
    validation_data=(X_test, y_test),
    class_weight=class_weights_dict
)

# Evaluate
preds = np.argmax(model.predict(X_test), axis=1)
true = np.argmax(y_test, axis=1)
acc = metrics.accuracy_score(true, preds)
print(f"\nFinal test accuracy: {acc * 100:.2f}%")
