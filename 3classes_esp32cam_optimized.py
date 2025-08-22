import cv2
import numpy as np
from ultralytics import YOLO
import requests
import time
import os
from datetime import datetime

# Configuration
MODEL_PATH = "best.pt"
ESP32_IP = "10.25.250.144"
ESP32_URL = f"http://{ESP32_IP}/cam-mid.jpg"

# Load model
model = YOLO(MODEL_PATH)

# Class names and colors
class_names = {0: 'car', 1: 'non-car', 2: 'pedestrian'}
class_colors = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0)}

# Create session for faster requests
session = requests.Session()

# Create directory for saved images if it doesn't exist
save_dir = "saved_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print("Starting ESP32-CAM Live Detection...")
print("Press 'q' to quit")
print("Press 's' to save current frame")

while True:
    try:
        # Get frame from ESP32-CAM
        response = session.get(ESP32_URL, timeout=2)
        img_array = np.frombuffer(response.content, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            print("Failed to get frame")
            continue

        # Run detection
        results = model(frame, conf=0.5)

        # Count objects
        counts = {'car': 0, 'non-car': 0, 'pedestrian': 0}

        # Draw detections
        if results[0].boxes is not None:
            for box in results[0].boxes:
                # Get detection info
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                class_id = int(box.cls.item())
                confidence = box.conf.item()

                if confidence > 0.5:
                    # Get class name and color
                    class_name = class_names[class_id]
                    color = class_colors[class_id]
                    counts[class_name] += 1

                    # Draw box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{class_name}: {confidence:.2f}',
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show counts on screen
        cv2.putText(frame, f'Cars: {counts["car"]}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f'Non-cars: {counts["non-car"]}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f'Pedestrians: {counts["pedestrian"]}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Show live video
        cv2.imshow('ESP32-CAM Live Detection', frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{save_dir}/detection_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Image saved: {filename}")

        # Print counts
        total = sum(counts.values())
        if total > 0:
            print(f"Cars: {counts['car']}, Non-cars: {counts['non-car']}, Pedestrians: {counts['pedestrian']}")

    except Exception as e:
        print(f"Error: {e}")
        time.sleep(0.033)

# Cleanup
cv2.destroyAllWindows()
print("Detection stopped")