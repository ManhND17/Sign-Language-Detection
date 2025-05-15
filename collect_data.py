import cv2
import numpy as np
import mediapipe as mp
import os
from utils import normalize_landmarks

label = input("Nhập tên ký hiệu (vd: A): ").strip().upper()
save_dir = "data"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            lm = []
            for pt in hand.landmark:
                lm.extend([pt.x, pt.y, pt.z])
            norm = normalize_landmarks(lm)
            filename = f"{label}_{count:03d}.npy"
            np.save(os.path.join(save_dir, filename), norm)
            print("Saved:", filename)
            count += 1

    cv2.putText(frame, f"Label: {label} | Saved: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Collect Data - Press 's' to save, 'Esc' to exit", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
