import mediapipe as mp
print("Trying to create Holistic object...")
with mp.solutions.holistic.Holistic(static_image_mode=True) as holistic:
    print("Holistic object created successfully!")