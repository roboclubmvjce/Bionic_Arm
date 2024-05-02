import cv2
import mediapipe as mp

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Initialize MediaPipe drawing utils and OpenCV window.
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)  # Use 0 for the webcam

finger_states = [0] * 5  # Initialize finger states array with zeros

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Flip the image horizontally for a later selfie-view display
        # Convert the BGR image to RGB.
        image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # Process the image and find hands
        results = hands.process(image_rgb)

        # Convert the image color back so it can be displayed
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get relevant landmark points for knuckles (MCPs)
                knuckle_indices = [17, 13, 9, 5]  # Pinky, Ring, Middle, Index MCPs
                
                # Determine finger openness based on knuckle positions
                for i, idx in enumerate(knuckle_indices):
                    # Pinky, Ring, Middle, Index finger MCP (knuckle) and PIP (second joint) indices
                    mcp_idx = idx
                    pip_idx = mcp_idx + 1

                    # Check if the finger is down or up based on the distance between MCP and PIP
                    if hand_landmarks.landmark[pip_idx].y > hand_landmarks.landmark[mcp_idx].y:
                        finger_states[i] = 1  # Finger down
                    else:
                        finger_states[i] = 0  # Finger up

                # Draw hand landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the finger states array on the image
        finger_states_text = " ".join(str(state) for state in finger_states)
        cv2.putText(image, f"Finger States: {finger_states_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the image
        cv2.imshow('MediaPipe Hands', image)
        
        # Break the loop if 'Esc' is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()