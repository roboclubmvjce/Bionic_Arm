import cv2
import mediapipe as mp
import math
import numpy as np
# Initialize MediaPipe Hands module.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Initialize MediaPipe drawing utils for drawing hand landmarks on the image.
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from the first webcam on your computer.
cap = cv2.VideoCapture(0)

def calculate_angle(a, b, c):
    """ Calculate angle between three points using the Law of Cosines. """
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

def is_knuckle_closed(hand_landmarks, knuckle_idx, tip_idx, threshold=20):
    knuckle_point = (hand_landmarks.landmark[knuckle_idx].x * image.shape[1], hand_landmarks.landmark[knuckle_idx].y * image.shape[0])
    tip_point = (hand_landmarks.landmark[tip_idx].x * image.shape[1], hand_landmarks.landmark[tip_idx].y * image.shape[0])
    
    distance = math.dist(knuckle_point, tip_point)
    if distance < threshold:
        return 1
    else:
        return 0

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)  # Process the image and detect hands

        # Convert the image color back so it can be displayed properly
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks.
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get relevant landmark points
                knuckle_indices = [1, 5, 9, 13, 17]  # Index, Middle, Ring, Pinky MCP
                tip_indices = [2, 6, 10, 14, 18]  # Index, Middle, Ring, Pinky Tip
                
                # Calculate and display angles for each finger
                for i, (knuckle_idx, tip_idx) in enumerate(zip(knuckle_indices, tip_indices)):
                    is_closed = is_knuckle_closed(hand_landmarks, knuckle_idx, tip_idx)
                    cv2.putText(image, f"{['Thumb','Index', 'Middle', 'Ring', 'Pinky'][i]} Knuckle Closed: {is_closed}", (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()