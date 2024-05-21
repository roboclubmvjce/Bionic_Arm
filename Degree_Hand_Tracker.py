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

# Get the width and height of the frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the bounding box for hand detection
box_width = int(width * 0.4)
box_height = int(height * 0.4)
box_x = (width - box_width) // 2
box_y = (height - box_height) // 2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break

    # Draw the bounding box
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 0, 0), 2)

    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image
    results = hands.process(frame_rgb)

    # Draw hand landmarks and analyze finger angles
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the center of the hand
            hand_center_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width)
            hand_center_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height)

            # Check if the center of the hand is within the bounding box
            if box_x < hand_center_x < box_x + box_width and box_y < hand_center_y < box_y + box_height:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get landmark positions
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append((lm.x, lm.y, lm.z))

                # Calculate finger bending status based on y coordinates
                thumb_bent = landmarks[4][1] < landmarks[3][1]
                index_bent = landmarks[8][1] > landmarks[6][1]
                middle_bent = landmarks[12][1] > landmarks[10][1]
                ring_bent = landmarks[16][1] > landmarks[14][1]
                pinky_bent = landmarks[20][1] > landmarks[18][1]

                # Calculate angles
                thumb_angle = calculate_angle(landmarks[2], landmarks[4], landmarks[3])
                index_angle = calculate_angle(landmarks[5], landmarks[6], landmarks[8])
                middle_angle = calculate_angle(landmarks[9], landmarks[10], landmarks[12])
                ring_angle = calculate_angle(landmarks[13], landmarks[14], landmarks[16])
                pinky_angle = calculate_angle(landmarks[17], landmarks[18], landmarks[20])

                # Display finger status on the image
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f'Thumb: {"Straight" if not thumb_bent else "Bent"} ({int(thumb_angle)} deg)', (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Index: {"Straight" if not index_bent else "Bent"} ({int(index_angle)} deg)', (10, 60), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Middle: {"Straight" if not middle_bent else "Bent"} ({int(middle_angle)} deg)', (10, 90), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Ring: {"Straight" if not ring_bent else "Bent"} ({int(ring_angle)} deg)', (10, 120), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Pinky: {"Straight" if not pinky_bent else "Bent"} ({int(pinky_angle)} deg)', (10, 150), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Hand Tracking', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
