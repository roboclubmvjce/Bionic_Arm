import cv2
import mediapipe as mp
import math

# Function to calculate angle between three points (p1, p2, p3)
def calculate_angle(p1, p2, p3):
    angle_rad = math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - math.atan2(p1[1] - p2[1], p1[0] - p2[0])
    angle_deg = math.degrees(angle_rad)
    return angle_deg + 360 if angle_deg < 0 else angle_deg

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Open default camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break

    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image
    results = hands.process(frame_rgb)

    # Draw hand landmarks and analyze finger angles
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append((lm.x, lm.y, lm.z))

            # Calculate finger bending status based on x and y coordinates
            thumb_bent = landmarks[4][0] < landmarks[3][0]
            index_bent = landmarks[8][1] < landmarks[6][1]
            middle_bent = landmarks[12][1] < landmarks[10][1]
            ring_bent = landmarks[16][1] < landmarks[14][1]
            pinky_bent = landmarks[20][1] < landmarks[18][1]

            # Display finger status on the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f'Thumb: {"Bent" if thumb_bent else "Straight"}', (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Index: {"Straight" if index_bent else "Bent"}', (10, 60), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Middle: {"Straight" if middle_bent else "Bent"}', (10, 90), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Ring: {"Straight" if ring_bent else "Bent"}', (10, 120), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Pinky: {"Straight" if pinky_bent else "Bent"}', (10, 150), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Hand Tracking', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
