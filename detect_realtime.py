import cv2
import numpy as np
import mediapipe as mp
import torch
from train_pytorch_advanced import AdvancedMLP
from sklearn.preprocessing import LabelEncoder
from utils import normalize_landmarks

# --- Load labels và model ---
labels = np.load("classes.npy")
num_classes = len(labels)

model = AdvancedMLP(63, num_classes=num_classes)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

le = LabelEncoder()
le.classes_ = labels

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        lm = []
        for pt in hand_landmarks.landmark:
            lm.extend([pt.x, pt.y, pt.z])

        x = normalize_landmarks(lm)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred = model(x)
            pred_probs = torch.softmax(pred, dim=1)
            pred_idx = pred_probs.argmax().item()
            confidence = pred_probs[0, pred_idx].item()
            label = le.inverse_transform([pred_idx])[0]

            # Chọn màu theo độ tin cậy
            if confidence >= 0.7:
                color = (0, 255, 0)     
            elif confidence >= 0.4:
                color = (0, 255, 255) 
            else:
                color = (0, 0, 255) 

            # Hiển thị label + confidence
            cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)

    cv2.imshow("Hand Sign Recognition", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
