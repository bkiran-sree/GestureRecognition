import cv2
import mediapipe as mp
import math

# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Function to detect 'OK' gesture
def is_okie(hand_landmarks):
    # Check if the thumb and index finger are touching
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Correct Euclidean distance formulas
    distance = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    return distance < 0.05  # Threshold for "touching"


# Function to detect 'Hungry' gesture
def is_hungry(hand_landmarks):
    distance_1 = math.sqrt((hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x -
                            hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x) ** 2)
    distance_2 = math.sqrt((hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x - hand_landmarks.landmark[
        mp_hands.HandLandmark.WRIST].x) ** 2)
    return distance_1 > 0.1 and distance_2 > 0.1


# Function to detect 'Angry' gesture (fist)
def is_angry(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    return (
                thumb_tip.y < index_tip.y and middle_tip.y < index_tip.y and ring_tip.y < index_tip.y and pinky_tip.y < index_tip.y)


# Function to detect 'Peace' gesture
def is_peace(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    return (index_tip.y < middle_tip.y)  # Both fingers extended in a peace gesture


# Function to detect 'Thumbs Up' gesture
def is_thumbs_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return (thumb_tip.y < index_tip.y)  # Thumb up gesture


# Function to detect 'Stop' gesture
def is_stop(hand_landmarks):
    distance_1 = math.sqrt((hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x - hand_landmarks.landmark[
        mp_hands.HandLandmark.PINKY_TIP].x) ** 2)
    distance_2 = math.sqrt((hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x - hand_landmarks.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_TIP].x) ** 2)
    return distance_1 > 0.1 and distance_2 > 0.1


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

            # Detect gestures
            if is_okie(hand_landmarks):
                cv2.putText(frame, "Gesture: OK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_hungry(hand_landmarks):
                cv2.putText(frame, "Gesture: Hungry", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_angry(hand_landmarks):
                cv2.putText(frame, "Gesture: Angry", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_peace(hand_landmarks):
                cv2.putText(frame, "Gesture: Peace", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_thumbs_up(hand_landmarks):
                cv2.putText(frame, "Gesture: Thumbs Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_stop(hand_landmarks):
                cv2.putText(frame, "Gesture: Stop", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()