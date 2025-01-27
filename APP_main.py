import os
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import mediapipe as mp

# 1. Load model (this part stays the same)
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

# 2. Define the dataset folder path
dataset_folder = r"C:\Users\bkira\OneDrive\Desktop\exeed\MP_Data"  

# 3. Load and process the dataset into sequences of 30 frames
def load_and_process_data(dataset_folder, sequence_length=30):
    files = os.listdir(dataset_folder)  # Get list of files in the dataset folder
    files.sort()  # Sort files if they are not in the right order (e.g., image_1.jpg, image_2.jpg, etc.)
    frames = []
    
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):  # Process only image files
            img = cv2.imread(os.path.join(dataset_folder, file))  # Read the image
            img = cv2.resize(img, (300, 400))  # Resize to match model input size (300x400 in your case)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frames.append(img)
    
    # Create sequences of 30 frames from the images
    sequences = [frames[i:i+sequence_length] for i in range(len(frames) - sequence_length + 1)]
    return np.array(sequences)

# 4. Load the dataset (call the function to load images from your dataset folder)
sequences = load_and_process_data(dataset_folder)

# 5. Define the actions list (you must have this list of actions)
actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
 # Replace with actual actions

# 6. Predict the action for each sequence
for sequence in sequences:
    res = model.predict(np.expand_dims(sequence, axis=0))[0]  # Shape: (1, num_actions)
    predicted_action = actions[np.argmax(res)]  # Get the predicted action (most likely class)
    
    # Print the predicted action
    print(predicted_action)

