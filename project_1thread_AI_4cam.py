import cv2
import numpy as np
from ultralytics import YOLO
import requests
import time
import os
import threading
import serial
import json
from datetime import datetime
from collections import defaultdict

# Configuration
MODEL_PATH = "best.pt"
ESP32_IPS = [
    "10.25.250.144",  # Lane 0 (North)
    "10.25.250.145",  # Lane 1 (South)
    "10.25.250.146",  # Lane 2 (East)
    "10.25.250.147"  # Lane 3 (West)
]

# Arduino communication
ARDUINO_PORT = "COM3"  # Change to your Arduino port (e.g., "/dev/ttyUSB0" on Linux)
BAUD_RATE = 9600

# Class names and priorities (higher number = higher priority)
class_names = {0: 'car', 1: 'emergency', 2: 'heavy', 3: 'pedestrian', 4: 'public'}
class_priorities = {0: 1, 1: 5, 2: 4, 3: 0, 4: 3}  # emergency=5, heavy=4, public=3, car=1, pedestrian=0
class_colors = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 255), 3: (255, 0, 0), 4: (0, 255, 255)}

# Traffic light timing (seconds)
DETECTION_INTERVAL = 2  # How often to send data to Arduino
GREEN_TIME_BASE = 10  # Base green light time
MAX_GREEN_TIME = 30  # Maximum green light time
YELLOW_TIME = 3  # Yellow light duration
RED_CLEARANCE = 2  # All red time between changes

# Global variables
running = True
arduino_connection = None
lane_data = defaultdict(lambda: defaultdict(int))  # lane_data[lane][class] = count
current_priority_lane = 0
last_decision_time = 0


class ArduinoController:
    def __init__(self):
        self.connection = None
        self.connect()

    def connect(self):
        """Connect to Arduino"""
        try:
            self.connection = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            print("Arduino connected successfully!")
            return True
        except Exception as e:
            print(f"Arduino connection failed: {e}")
            return False

    def send_data(self, lane_counts, priority_lane):
        """Send detection data to Arduino"""
        if not self.connection:
            return False

        try:
            # Create data packet
            data = {
                "lanes": lane_counts,
                "priority_lane": priority_lane,
                "timestamp": int(time.time())
            }

            # Send JSON data
            json_data = json.dumps(data) + "\n"
            self.connection.write(json_data.encode())
            print(f"Sent to Arduino: Priority Lane {priority_lane}")
            return True

        except Exception as e:
            print(f"Arduino send error: {e}")
            return False

    def close(self):
        if self.connection:
            self.connection.close()


class CameraThread:
    def __init__(self, camera_id, ip_address):
        self.camera_id = camera_id
        self.ip_address = ip_address
        self.url = f"http://{ip_address}/cam-fast.jpg"
        self.session = requests.Session()
        self.model = None
        self.current_frame = None
        self.is_active = False
        self.failed_attempts = 0
        self.max_attempts = 10
        self.fps = 0
        self.frame_time = time.time()

        # Create save directory
        self.save_dir = f"saved_images/camera_{camera_id}"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def test_connection(self):
        """Test if camera is accessible"""
        try:
            print(f"Testing camera {self.camera_id} ({self.ip_address})...")
            response = self.session.get(self.url, timeout=3)
            if response.status_code == 200 and len(response.content) > 1000:
                print(f"Camera {self.camera_id} connected successfully!")
                return True
            else:
                print(f"Camera {self.camera_id} - Invalid response")
                return False
        except Exception as e:
            print(f"Camera {self.camera_id} - Connection failed: {e}")
            return False

    def calculate_fps(self):
        """Calculate FPS"""
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.frame_time)
        self.frame_time = current_time

    def run(self):
        """Main thread function"""
        global lane_data

        print(f"Starting camera {self.camera_id} thread for IP: {self.ip_address}")

        # Initial connection test
        for attempt in range(3):
            if self.test_connection():
                break
            print(f"Camera {self.camera_id} - Attempt {attempt + 1} failed, retrying...")
            time.sleep(3)
        else:
            print(f"Camera {self.camera_id} - All connection attempts failed. Stopping thread.")
            return

        # Load AI model
        if self.failed_attempts < self.max_attempts:
            print(f"Loading AI model for camera {self.camera_id}...")
            self.model = YOLO(MODEL_PATH)
            self.is_active = True
            print(f"Camera {self.camera_id} ready!")

        while running and self.is_active and self.failed_attempts < self.max_attempts:
            try:
                # Get frame from ESP32-CAM
                response = self.session.get(self.url, timeout=5)

                if response.status_code != 200:
                    print(f"Camera {self.camera_id} - HTTP error: {response.status_code}")
                    self.failed_attempts += 1
                    time.sleep(2)
                    continue

                img_array = np.frombuffer(response.content, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if frame is None:
                    print(f"Camera {self.camera_id} - Failed to decode frame")
                    self.failed_attempts += 1
                    time.sleep(2)
                    continue

                self.failed_attempts = 0
                self.current_frame = frame.copy()
                self.calculate_fps()

                # Run AI detection
                results = self.model(frame, conf=0.5)

                # Count objects by class
                counts = {class_name: 0 for class_name in class_names.values()}

                # Draw detections and count
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        class_id = int(box.cls.item())
                        confidence = box.conf.item()

                        if confidence > 0.5 and class_id in class_names:
                            class_name = class_names[class_id]
                            color = class_colors[class_id]
                            counts[class_name] += 1

                            # Draw box and label
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f'{class_name}: {confidence:.2f}',
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Update global lane data
                for class_name, count in counts.items():
                    lane_data[self.camera_id][class_name] = count

                # Display information on frame
                y_pos = 20
                for class_name, count in counts.items():
                    color = class_colors[list(class_names.values()).index(class_name)]
                    cv2.putText(frame, f'{class_name}: {count}', (5, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_pos += 25

                cv2.putText(frame, f'FPS: {self.fps:.1f}', (5, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f'Lane {self.camera_id} - {self.ip_address}',
                            (5, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Show video
                cv2.imshow(f'Lane {self.camera_id} - ESP32-CAM', frame)
                cv2.waitKey(1)

                time.sleep(0.1)  # Control frame rate

            except Exception as e:
                print(f"Camera {self.camera_id} error: {e}")
                self.failed_attempts += 1
                time.sleep(2)

        print(f"Camera {self.camera_id} - Thread ended")

    def save_frame(self):
        """Save current frame"""
        if self.current_frame is not None and self.is_active:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.save_dir}/detection_cam{self.camera_id}_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            print(f"Camera {self.camera_id}: Saved {filename}")
            return True
        return False


def calculate_lane_priority(lane_counts):
    """Calculate priority score for each lane"""
    priorities = {}

    for lane_id, counts in lane_counts.items():
        score = 0
        total_vehicles = 0

        for class_name, count in counts.items():
            if count > 0:
                class_id = list(class_names.values()).index(class_name)
                priority = class_priorities[class_id]

                # Emergency vehicles get maximum priority
                if class_name == 'emergency' and count > 0:
                    score += 1000 * count
                # Heavy vehicles get high priority
                elif class_name == 'heavy' and count > 0:
                    score += 100 * count
                # Other vehicles
                else:
                    score += priority * count

                if class_name != 'pedestrian':
                    total_vehicles += count

        # Boost score based on total vehicles waiting
        if total_vehicles > 5:
            score *= 1.5
        elif total_vehicles > 10:
            score *= 2.0

        priorities[lane_id] = score

    return priorities


def traffic_control_loop():
    """Main traffic control decision loop"""
    global current_priority_lane, last_decision_time, arduino_connection

    while running:
        try:
            current_time = time.time()

            # Make decisions every DETECTION_INTERVAL seconds
            if current_time - last_decision_time >= DETECTION_INTERVAL:
                # Calculate priorities for each lane
                lane_priorities = calculate_lane_priority(lane_data)

                # Find lane with highest priority
                if lane_priorities:
                    new_priority_lane = max(lane_priorities, key=lane_priorities.get)
                    max_priority = lane_priorities[new_priority_lane]

                    # Only change if there's significant traffic or emergency
                    if max_priority > 0 and (new_priority_lane != current_priority_lane or max_priority >= 100):
                        current_priority_lane = new_priority_lane

                        # Send data to Arduino
                        if arduino_connection:
                            lane_counts_dict = dict(lane_data)
                            arduino_connection.send_data(lane_counts_dict, current_priority_lane)

                        print(f"Priority Lane: {current_priority_lane}, Score: {max_priority:.1f}")
                        for lane_id, counts in lane_data.items():
                            if any(count > 0 for count in counts.values()):
                                print(f"  Lane {lane_id}: {dict(counts)}")

                last_decision_time = current_time

            time.sleep(0.5)

        except Exception as e:
            print(f"Traffic control error: {e}")
            time.sleep(1)


def main():
    global running, arduino_connection, shared_model

    print("Starting AI-Driven Traffic Light Management System...")
    print("Lane Layout: 0=North, 1=South, 2=East, 3=West")
    print("Priority: Emergency > Heavy > Public > Car > Pedestrian")
    print("Press 'q' to quit, 's' to save all frames")
    print()

    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Using device: {DEVICE}")
    else:
        print("GPU not available, using CPU")
        DEVICE = "cpu"

    # Load shared AI model once
    print("Loading AI model...")
    shared_model = YOLO(MODEL_PATH)

    # Explicitly set device and optimize for inference
    shared_model.to(DEVICE)
    if DEVICE == "cuda":
        # Enable mixed precision for faster inference
        shared_model.model.half()  # Use FP16 for faster GPU inference
        print("GPU optimization enabled (FP16)")

    print(f"AI model loaded successfully on {DEVICE}!")

    # Initialize Arduino connection
    arduino_connection = ArduinoController()

    # Create and start camera threads
    cameras = []
    threads = []

    for i, ip in enumerate(ESP32_IPS):
        camera = CameraThread(i, ip)
        cameras.append(camera)

        thread = threading.Thread(target=camera.run, daemon=True)
        thread.start()
        threads.append(thread)

    # Start traffic control thread
    control_thread = threading.Thread(target=traffic_control_loop, daemon=True)
    control_thread.start()

    # Give threads time to initialize
    time.sleep(5)

    try:
        while True:
            key = cv2.waitKey(100) & 0xFF

            if key == ord('q'):
                print("Shutting down...")
                break
            elif key == ord('s'):
                # Save all frames
                saved = 0
                for camera in cameras:
                    if camera.save_frame():
                        saved += 1
                print(f"Saved frames from {saved} active cameras")
            elif key == ord('g'):
                # Print GPU memory usage
                if torch.cuda.is_available():
                    print(
                        f"GPU Memory: {torch.cuda.memory_allocated() / 1024 ** 3:.1f}GB / {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f}GB")

            # Check if any cameras are still active
            active_cameras = sum(1 for cam in cameras if cam.is_active)
            if active_cameras == 0:
                print("No active cameras. Exiting...")
                break

    except KeyboardInterrupt:
        print("\nKeyboard interrupt")

    finally:
        running = False
        if arduino_connection:
            arduino_connection.close()
        time.sleep(2)
        cv2.destroyAllWindows()
        print("System stopped")


if __name__ == "__main__":
    main()
