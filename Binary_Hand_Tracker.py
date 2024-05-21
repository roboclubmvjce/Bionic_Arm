import cv2
import mediapipe as mp
import math
import serial
import serial.tools.list_ports

# Function to calculate angle between three points (p1, p2, p3)
def calculate_angle(p1, p2, p3):
    angle_rad = math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - math.atan2(p1[1] - p2[1], p1[0] - p2[0])
    angle_deg = math.degrees(angle_rad)
    return angle_deg + 360 if angle_deg < 0 else angle_deg

# Function to update hand state array
def update_hand_state(thumb_bent, index_bent, middle_bent, ring_bent, pinky_bent):
    hand_state = [int(thumb_bent), int(index_bent), int(middle_bent), int(ring_bent), int(pinky_bent)]
    return hand_state

# Set up serial communication
ports = serial.tools.list_ports.comports()
serial_inst = serial.Serial()

ports_list = []
for port in ports:
    ports_list.append(str(port))
    print(str(port))

val = input('Select Port: COM')
for i in range(len(ports_list)):
    if ports_list[i].startswith(f'COM{val}'):
        port_var = f'COM{val}'
        print(port_var)

serial_inst.baudrate = 9600
serial_inst.port = port_var
serial_inst.open()

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

    # Get the dimensions of the frame
    height, width, _ = frame.shape

    # Define the box dimensions
    box_left = width // 4
    box_top = height // 4
    box_right = 3 * width // 4
    box_bottom = 3 * height // 4

    # Draw a box in the middle of the frame
    cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), (255, 0, 0), 2)

    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image
    results = hands.process(frame_rgb)

    # Draw hand landmarks and analyze finger angles
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the landmark positions
            landmarks = []
            for lm in hand_landmarks.landmark:
                x = int(lm.x * width)
                y = int(lm.y * height)
                landmarks.append((x, y))

            # Check if the hand is within the box
            hand_x, hand_y = landmarks[0]  # Using the position of the wrist
            if box_left < hand_x < box_right and box_top < hand_y < box_bottom:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Calculate finger bending status based on x and y coordinates
                thumb_bent = landmarks[4][0] < landmarks[3][0]
                index_bent = landmarks[8][1] < landmarks[6][1]
                middle_bent = landmarks[12][1] < landmarks[10][1]
                ring_bent = landmarks[16][1] < landmarks[14][1]
                pinky_bent = landmarks[20][1] < landmarks[18][1]

                # Update hand state array
                hand_state = update_hand_state(thumb_bent, index_bent, middle_bent, ring_bent, pinky_bent)

                # Send hand state to Arduino
                hand_state_str = ''.join(map(str, hand_state)) + '\n'
                serial_inst.write(hand_state_str.encode('utf-8'))

                # Display finger status on the image
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f'Thumb: {"Bent" if thumb_bent else "Straight"}', (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Index: {"Bent" if index_bent else "Straight"}', (10, 60), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Middle: {"Bent" if middle_bent else "Straight"}', (10, 90), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Ring: {"Bent" if ring_bent else "Straight"}', (10, 120), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Pinky: {"Bent" if pinky_bent else "Straight"}', (10, 150), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                # Display the hand state array
                cv2.putText(frame, f'Hand State: {hand_state}', (10, 180), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Hand Tracking', frame)

    # Exit when 'ESC' is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
serial_inst.close()
