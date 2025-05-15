import cv2
import mediapipe as mp

image_path = "images/A-samples/0.jpg"  # thay bằng 1 ảnh cụ thể
image = cv2.imread(image_path)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = hands.process(image_rgb)

if results.multi_hand_landmarks:
    print("✅ Tay đã được phát hiện")
else:
    print("❌ KHÔNG phát hiện tay")
