import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    canvas = None
    prev_x, prev_y = None, None 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
                index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
                middle_x, middle_y = int(middle_tip.x * frame.shape[1]), int(middle_tip.y * frame.shape[0])

                if abs(thumb_x - index_x) < 20 and abs(thumb_y - index_y) < 20:
                    if canvas is None:
                        canvas = np.zeros_like(image)
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), (0, 0, 255), 3)
                    prev_x, prev_y = index_x, index_y
                else:
                    prev_x, prev_y = None, None


                if middle_y > thumb_y:
                    if canvas is not None:
                        cv2.circle(canvas, (middle_x, middle_y), 30, (0, 0, 0), -1)


        if canvas is not None:
            image = cv2.add(image, canvas)


        cv2.imshow('Hand Tracking', image)


        if cv2.waitKey(1) & 0xFF == ord('c'):
            canvas = None

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
