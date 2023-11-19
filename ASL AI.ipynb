import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2)

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

new_width = 640
new_height = 480

def is_thumbs_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    if thumb_tip.y < index_tip.y < middle_tip.y < ring_tip.y < pinky_tip.y:
        return True
    else:
        return False

def is_index_up(hand_landmarks):    
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP] #
    index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP] # 
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    
    if index_tip.y < index_dip.y < index_pip.y < index_mcp.y:
        return True
    else:
        return False

while True:
    ret, frame = cap.read()

    # Resize the frame
    frame = cv2.resize(frame, (new_width, new_height))

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    landmarks_frame = np.zeros_like(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                landmarks_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_thumbs_up(hand_landmarks):
                cv2.putText(frame, 'Thumbs up!', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 255, 12), 2,
                            cv2.LINE_AA)
            elif is_index_up(hand_landmarks):
                cv2.putText(frame, 'Index!', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 255, 12), 2, cv2.LINE_AA)

    frame = cv2.addWeighted(frame, 1, landmarks_frame, 1, 0)
    cv2.imshow('MediaPipe Hands', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
