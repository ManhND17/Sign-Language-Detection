import numpy as np

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    center = landmarks[0]  # cổ tay làm gốc
    landmarks -= center  # đưa về gốc toạ độ cổ tay
    max_val = np.max(np.linalg.norm(landmarks, axis=1))
    landmarks /= max_val  # scale về [0, 1]
    return landmarks.flatten()