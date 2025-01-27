import cv2
import mediapipe as mp
import math

# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Function to detect letter A gesture (Thumb and index finger extended)
def is_letter_A(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return thumb_tip.y < index_tip.y  # Thumb and index extended, other fingers curled

# Function to detect letter B gesture
def is_letter_B(hand_landmarks):
    # Example: Fist with index finger pointing upwards (thumb and other fingers curled)
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return index_tip.y < 0.5  # Extended index

# Function to detect letter C gesture
def is_letter_C(hand_landmarks):
    # "C" shaped gesture with thumb and fingers
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return thumb_tip.x > index_tip.x  # Forming "C" shape

# Function to detect letter D gesture
def is_letter_D(hand_landmarks):
    # Example: "D" shaped gesture, thumb and index extended, other fingers curled
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return index_tip.y < 0.5  # Index extended

# Continue similarly for other letters...

# Function to detect letter Z gesture (index and pinky extended forming a "Z")
def is_letter_Z(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    return index_tip.y < 0.5 and pinky_tip.y < 0.5  # Index and pinky extended, other fingers curled

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

            # Detect letter gestures
            if is_letter_A(hand_landmarks):
                cv2.putText(frame, "Gesture: A", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_letter_B(hand_landmarks):
                cv2.putText(frame, "Gesture: B", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_letter_C(hand_landmarks):
                cv2.putText(frame, "Gesture: C", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_letter_D(hand_landmarks):
                cv2.putText(frame, "Gesture: D", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Continue adding conditions for other letters...
            

            # For letter Z (index and pinky extended)
            elif is_letter_Z(hand_landmarks):
                cv2.putText(frame, "Gesture: Z", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
