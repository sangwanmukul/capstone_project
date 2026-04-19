import cv2
import numpy as np
import random
import string
import os

# =========================
# 1. Random Text Generator
# =========================
def random_text(length=5):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


# =========================
# 2. CAPTCHA Generator (STRONG)
# =========================
def generate_captcha(text):
    img = np.ones((80, 200, 3), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Random placement
    x = random.randint(10, 40)
    y = random.randint(40, 70)

    # Draw characters individually (better distortion)
    for i, ch in enumerate(text):
        offset_x = x + i * 30 + random.randint(-5, 5)
        offset_y = y + random.randint(-5, 5)
        cv2.putText(img, ch, (offset_x, offset_y),
                    font, 1.0, (0, 0, 0), 2)

    # Noise
    for _ in range(300):
        x_noise = random.randint(0, 199)
        y_noise = random.randint(0, 79)
        img[y_noise, x_noise] = np.random.randint(0, 255, 3)

    # Random lines
    for _ in range(5):
        x1, y1 = random.randint(0, 200), random.randint(0, 80)
        x2, y2 = random.randint(0, 200), random.randint(0, 80)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 1)

    # Rotation
    angle = random.randint(-30, 30)
    M = cv2.getRotationMatrix2D((100, 40), angle, 1)
    img = cv2.warpAffine(img, M, (200, 80), borderValue=(255, 255, 255))

    # Wave distortion
    for i in range(img.shape[0]):
        shift = int(5 * np.sin(i / 8.0))
        img[i] = np.roll(img[i], shift, axis=0)

    # Blur
    img = cv2.GaussianBlur(img, (5, 5), 0)

    return img


# =========================
# 3. Save Captchas (WITH LABELS 🔥)
# =========================
def generate_and_save_captchas(n=20, folder="real_captcha_samples"):
    os.makedirs(folder, exist_ok=True)

    # Clear old files (IMPORTANT)
    for f in os.listdir(folder):
        os.remove(os.path.join(folder, f))

    for i in range(n):
        text = random_text()
        img = generate_captcha(text)

        # ✅ Save WITH ground truth text
        path = os.path.join(folder, f"captcha_{text}.png")
        cv2.imwrite(path, img)

    print(f"[INFO] Generated {n} CAPTCHA images → {folder}")