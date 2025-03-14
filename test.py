import cv2
import mediapipe as mp

def get_finger_status(hand_landmarks, hand_label, frame_width, frame_height):
    # Convert normalized landmarks to pixel coordinates
    landmarks = [
        (int(lm.x * frame_width), int(lm.y * frame_height))
        for lm in hand_landmarks.landmark
    ]
    
    # For index, middle, ring, and pinky, compare tip and pip y-coordinates
    index_extended = landmarks[8][1] < landmarks[6][1]
    middle_extended = landmarks[12][1] < landmarks[10][1]
    ring_extended = landmarks[16][1] < landmarks[14][1]
    pinky_extended = landmarks[20][1] < landmarks[18][1]
    
    # For thumb, we compare x-coordinates. Note that if the image is flipped,
    # the logic might need adjustment. Using handedness information to decide.
    if hand_label == "Right":
        thumb_extended = landmarks[4][0] > landmarks[3][0]
    else:
        thumb_extended = landmarks[4][0] < landmarks[3][0]
    
    return [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]

def recognize_gesture(fingers):
    # fingers is a list of booleans: [thumb, index, middle, ring, pinky]
    # Gesture recognition based on simple heuristics
    if all(not f for f in fingers):
        return "Fist"
    elif all(fingers):
        return "Open Hand"
    elif fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
        return "Peace Sign"
    elif fingers[0] and not any(fingers[1:]):
        return "Thumbs Up"
    else:
        return "Unknown"

def main():
    cap = cv2.VideoCapture(0)
    
    # Initialize MediaPipe Hands.
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip the frame horizontally for a natural mirror view
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and get hand landmarks
        result = hands.process(frame_rgb)
        frame_height, frame_width, _ = frame.shape
        
        if result.multi_hand_landmarks and result.multi_handedness:
            # Iterate through each detected hand and its handedness
            for hand_landmarks, hand_handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                # Draw hand landmarks on the frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get the handedness label ("Left" or "Right")
                hand_label = hand_handedness.classification[0].label
                
                # Get the status of each finger
                fingers = get_finger_status(hand_landmarks, hand_label, frame_width, frame_height)
                
                # Recognize the gesture based on finger status
                gesture = recognize_gesture(fingers)
                
                # Get the wrist coordinates to position the gesture text
                wrist_x = int(hand_landmarks.landmark[0].x * frame_width)
                wrist_y = int(hand_landmarks.landmark[0].y * frame_height)
                
                # Display the recognized gesture on the frame
                cv2.putText(frame, gesture, (wrist_x - 30, wrist_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Gesture Recognition", frame)
        
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
