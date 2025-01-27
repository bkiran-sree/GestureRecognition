import cv2
import mediapipe as mp
import math

# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Function to detect number 1 gesture (index finger extended)
def is_number_1(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return index_tip.y < 0.5  # Only index finger extended

# Function to detect number 2 gesture (index and middle fingers extended)
def is_number_2(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    return index_tip.y < 0.5 and middle_tip.y < 0.5

# Function to detect number 3 gesture (index, middle, and ring fingers extended)
def is_number_3(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    return index_tip.y < 0.5 and middle_tip.y < 0.5 and ring_tip.y < 0.5

# Function to detect number 4 gesture (index, middle, ring, and pinky fingers extended)
def is_number_4(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    return index_tip.y < 0.5 and middle_tip.y < 0.5 and ring_tip.y < 0.5 and pinky_tip.y < 0.5

# Function to detect number 5 gesture (all fingers extended)
def is_number_5(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    return index_tip.y < 0.5 and middle_tip.y < 0.5 and ring_tip.y < 0.5 and pinky_tip.y < 0.5

# Function to detect number 6 gesture (thumb, index, middle, and ring fingers extended)
def is_number_6(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    return thumb_tip.y < 0.5 and index_tip.y < 0.5 and middle_tip.y < 0.5 and ring_tip.y < 0.5

# Function to detect number 7 gesture (thumb, index, and pinky fingers extended)
def is_number_7(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    return thumb_tip.y < 0.5 and index_tip.y < 0.5 and pinky_tip.y < 0.5

# Function to detect number 8 gesture (all fingers extended with the pinky slightly bent)
def is_number_8(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    return index_tip.y < 0.5 and middle_tip.y < 0.5 and ring_tip.y < 0.5 and pinky_tip.y < 0.5

# Function to detect number 9 gesture (index, middle, ring, and pinky fingers extended with thumb bent)
def is_number_9(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    return index_tip.y < 0.5 and middle_tip.y < 0.5 and ring_tip.y < 0.5 and pinky_tip.y < 0.5 and thumb_tip.y > 0.5

# Function to detect number 0 gesture (fist with thumb out)
def is_number_0(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return index_tip.y > 0.5 and thumb_tip.y < 0.5

# Main loop to detect gestures
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the image horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to RGB (MediaPipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to get hand landmarks
    results = hands.process(rgb_frame)

    # Check if hands are found
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect number gestures
            if is_number_0(hand_landmarks):
                cv2.putText(frame, "Gesture: 0", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_number_1(hand_landmarks):
                cv2.putText(frame, "Gesture: 1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_number_2(hand_landmarks):
                cv2.putText(frame, "Gesture: 2", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_number_3(hand_landmarks):
                cv2.putText(frame, "Gesture: 3", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_number_4(hand_landmarks):
                cv2.putText(frame, "Gesture: 4", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_number_5(hand_landmarks):
                cv2.putText(frame, "Gesture: 5", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_number_6(hand_landmarks):
                cv2.putText(frame, "Gesture: 6", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_number_7(hand_landmarks):
                cv2.putText(frame, "Gesture: 7", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_number_8(hand_landmarks):
                cv2.putText(frame, "Gesture: 8", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_number_9(hand_landmarks):
                cv2.putText(frame, "Gesture: 9", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
