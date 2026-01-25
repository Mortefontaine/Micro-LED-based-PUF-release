import os
import cv2
import numpy as np
from ultralytics import YOLO

# === Parameter configuration ===
model_path = r"runs/detect/train/weights/best.pt"
input_root = r"D:\Research\PUF_micro-LED\videos3\frames_augment"
output_root = r"D:\Research\PUF_micro-LED\videos3\frames_selected_by_YOLO_aug"
crop_size = 256

# Load YOLO model
model = YOLO(model_path)
print("✅ YOLO model loaded")

# === Iterate over all images ===
for root, dirs, files in os.walk(input_root):
    rel_path = os.path.relpath(root, input_root)
    save_dir = os.path.join(output_root, rel_path)
    os.makedirs(save_dir, exist_ok=True)

    for file in files:
        if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        img_path = os.path.join(root, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠ Unable to read image: {img_path}")
            continue

        # YOLO inference
        results = model(img, conf=0.25)
        boxes = results[0].boxes.xywh.cpu().numpy()
        if len(boxes) == 0:
            print(f"⚠ No target detected: {img_path}")
            continue

        # Crop using the first detected target
        x_center, y_center, w, h = boxes[0]
        side = max(w, h, crop_size)

        x1 = int(x_center - side/2)
        y1 = int(y_center - side/2)
        x2 = int(x_center + side/2)
        y2 = int(y_center + side/2)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)

        cropped = img[y1:y2, x1:x2]

        # Resize to a uniform size
        resized = cv2.resize(cropped, (crop_size, crop_size), interpolation=cv2.INTER_AREA)

        # Save
        save_path = os.path.join(save_dir, file)
        cv2.imwrite(save_path, resized)
        print(f"✅ Saved: {save_path}")

print("🎉 All images processed!")
