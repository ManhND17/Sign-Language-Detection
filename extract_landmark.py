import os
import cv2
import numpy as np
import mediapipe as mp
from utils import normalize_landmarks

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

data_root = "images"
save_dir = "data"
os.makedirs(save_dir, exist_ok=True)

for folder in os.listdir(data_root):
    class_dir = os.path.join(data_root, folder)
    if not os.path.isdir(class_dir):
        continue

    # Tách nhãn từ tên thư mục, ví dụ: "A-samples" → "A"
    label = folder.split("-")[0].upper()

    # Tạo thư mục con trong data theo nhãn
    label_dir = os.path.join(save_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    for file in os.listdir(class_dir):
        if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        image_path = os.path.join(class_dir, file)
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Không đọc được ảnh:", image_path)
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            norm = normalize_landmarks(landmarks)

            filename = f"{label}_{os.path.splitext(file)[0]}.npy"
            path_save = os.path.join(label_dir, filename)  # Lưu trong thư mục con
            np.save(path_save, norm)
            print("✔️ Đã lưu:", path_save)
        else:
            print("⚠️ Không phát hiện tay:", image_path)
