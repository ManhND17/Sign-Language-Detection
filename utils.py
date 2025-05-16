import numpy as np

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    center = landmarks[0] 
    landmarks -= center  
    max_val = np.max(np.linalg.norm(landmarks, axis=1))
    landmarks /= max_val  
    return landmarks.flatten()