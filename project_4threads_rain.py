import cv2
import numpy as np
from ultralytics import YOLO
import requests
import threading
import time
import os
import serial
import json
from datetime import datetime
import torch

# ==============================
# Configuration
# ==============================
# ESP32 cameras configuration (4 lanes: North, East, South, West)
CAMERAS = [
    {"name": "North_Lane", "ip": "10.148.188.144", "model_path": "best_old.pt", "window_pos": (0, 0), "lane_id": 0},
    {"name": "East_Lane", "ip": "10.210.80.172", "model_path": "best_old.pt", "window_pos": (640, 0), "lane_id": 1},
    {"name": "South_Lane", "ip": "10.210.80.74", "model_path": "best_old.pt", "window_pos": (0, 480), "lane_id": 2},
    {"name": "West_Lane", "ip": "10.210.80.144", "model_path": "best_old.pt", "window_pos": (640, 480), "lane_id": 3}
]

# Serial Configuration
SERIAL_PORT = "COM9"  # Change to your Arduino port
SERIAL_BAUDRATE = 9600

# Global configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Updated class names and colors
class_names = {0: 'car', 1: 'emergency', 2: 'heavy', 3: 'pedestrian', 4: 'public'}
class_colors = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 165, 0), 3: (255, 0, 0), 4: (128, 0, 128)}

# Priority weights for each class
PRIORITY_WEIGHTS = {
    'emergency': 1000,  # Highest priority - immediate action
    'heavy': 80,
    'public': 60,
    'car': 50,
    'pedestrian': 40
}

# Create directory for saved images
save_dir = "saved_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print("Starting AI Traffic Management System...")
print("Press 'q' on any window to quit all cameras")
print("Press 's' on focused window to save current frame from that camera")

# ==============================
# Traffic Priority Manager
# ==============================
class TrafficPriorityManager:
    def __init__(self, serial_connection):
        self.serial_conn = serial_connection
        self.lane_data = [{'car': 0, 'emergency': 0, 'heavy': 0, 'pedestrian': 0, 'public': 0} for _ in range(4)]
        self.current_priority_lane = 0
        self.previous_priority_lane = -1  # Track previous priority lane
        self.last_update_time = time.time()
        self.update_interval = 4.0  # Update every 4 seconds

    def update_lane_data(self, lane_id, detections):
        """Update detection data for a specific lane"""
        self.lane_data[lane_id] = detections.copy()

    def calculate_priority_score(self, lane_data):
        """Calculate priority score for a lane based on detections"""
        score = 0
        breakdown = {}

        for vehicle_type, count in lane_data.items():
            if count > 0:
                weight = PRIORITY_WEIGHTS.get(vehicle_type, 0)
                contribution = weight * count
                score += contribution
                breakdown[vehicle_type] = f"{count} × {weight} = {contribution}"

        return score, breakdown

    def determine_priority_lane(self):
        """Determine which lane should have priority"""
        current_time = time.time()

        # Only update if enough time has passed
        if current_time - self.last_update_time < self.update_interval:
            return

        print(f"\n{'=' * 60}")
        print(f"TRAFFIC PRIORITY ANALYSIS - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'=' * 60}")

        lane_scores = []
        emergency_detected = False
        emergency_lane = -1
        pedestrians_detected = False
        pedestrian_lane = -1

        # Calculate scores for all lanes
        for i, lane_data in enumerate(self.lane_data):
            score, breakdown = self.calculate_priority_score(lane_data)
            lane_scores.append(score)

            lane_names = ["North", "East", "South", "West"]
            print(f"{lane_names[i]} Lane (ID: {i}):")
            print(f"  Detections: {lane_data}")
            print(f"  Score Breakdown: {breakdown if breakdown else 'No vehicles detected'}")
            print(f"  Total Score: {score}")

            # Check for emergency vehicles (highest priority)
            if lane_data.get('emergency', 0) > 0:
                emergency_detected = True
                emergency_lane = i
                print(f"  🚨 EMERGENCY VEHICLE DETECTED! 🚨")

            # Check for pedestrians
            if lane_data.get('pedestrian', 0) > 0:
                pedestrians_detected = True
                pedestrian_lane = i
                print(f"  🚶 Pedestrian detected")

            print()

        # Store previous priority lane
        self.previous_priority_lane = self.current_priority_lane

        # Determine priority based on conditions (Emergency has highest priority)
        if emergency_detected:
            priority_lane = emergency_lane
            priority_type = "emergency"
            print(f"PRIORITY DECISION: EMERGENCY")
            print(f"  Lane: {['North', 'East', 'South', 'West'][priority_lane]} (ID: {priority_lane})")
        elif pedestrians_detected:
            # For pedestrian safety, we send the lane where pedestrian is detected
            # Arduino will handle making that lane yellow first, then all red
            priority_lane = pedestrian_lane
            priority_type = "pedestrian_safety"
            print(f"PRIORITY DECISION: PEDESTRIAN SAFETY")
            print(f"  Pedestrian Lane: {['North', 'East', 'South', 'West'][priority_lane]} (ID: {priority_lane})")
            print(f"  All lanes will go RED for pedestrian crossing")
        else:
            # Normal traffic flow - highest score wins
            priority_lane = lane_scores.index(max(lane_scores))
            priority_type = "normal"
            print(f"PRIORITY DECISION: NORMAL TRAFFIC")
            print(f"  Lane: {['North', 'East', 'South', 'West'][priority_lane]} (ID: {priority_lane})")
            print(f"  Score: {lane_scores[priority_lane]}")

        print(f"  Previous Priority Lane: {self.previous_priority_lane}")

        # Send data to Arduino
        self.send_traffic_data(priority_lane, priority_type)
        self.current_priority_lane = priority_lane
        self.last_update_time = current_time
        print(f"{'=' * 60}\n")

    def send_traffic_data(self, priority_lane, priority_type):
        """Send traffic data to Arduino via serial"""
        try:
            traffic_data = {
                "priority_lane": priority_lane,
                "previous_priority_lane": self.previous_priority_lane,
                "priority_type": priority_type,
                "timestamp": int(time.time())
            }

            json_data = json.dumps(traffic_data)
            self.serial_conn.write((json_data + '\n').encode())

            print(f"📤 Sent to Arduino:")
            print(f"   Priority Lane: {priority_lane} ({priority_type})")
            print(f"   Previous Lane: {self.previous_priority_lane}")

        except Exception as e:
            print(f"❌ Serial communication error: {e}")

# ==============================
# Camera Detection Thread Class
# ==============================
class CameraDetector:
    def __init__(self, camera_config, stop_event, priority_manager):
        self.name = camera_config["name"]
        self.ip = camera_config["ip"]
        self.url = f"http://{self.ip}/cam-lo.jpg"
        self.model_path = camera_config["model_path"]
        self.window_pos = camera_config["window_pos"]
        self.lane_id = camera_config["lane_id"]
        self.stop_event = stop_event
        self.priority_manager = priority_manager

        # Load YOLO model
        self.model = YOLO(self.model_path)
        if device == "cuda":
            self.model.to(device).half()
        else:
            self.model.to(device)

        # Frame management
        self.latest_frame = None
        self.processed_frame = None
        self.lock = threading.Lock()

        # Statistics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        # Session for HTTP requests
        self.session = requests.Session()

        print(f"[{self.name}] Initialized on {self.ip}")

    def grab_and_process_frames(self):
        """Frame grabbing and processing"""
        while not self.stop_event.is_set():
            try:
                # Grab frame from ESP32
                response = self.session.get(self.url, timeout=2)
                if response.status_code == 200:
                    img_array = np.frombuffer(response.content, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    if frame is not None:
                        # Store original frame
                        with self.lock:
                            self.latest_frame = frame.copy()

                        # Run detection
                        start = time.time()
                        results = self.model(frame, conf=0.0, imgsz=320, verbose=False)
                        end = time.time()
                        inference_time = (end - start) * 1000

                        # Count objects
                        counts = {'car': 0, 'emergency': 0, 'heavy': 0, 'pedestrian': 0, 'public': 0}

                        # Draw detections
                        if results[0].boxes is not None:
                            for box in results[0].boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                class_id = int(box.cls.item())
                                confidence = box.conf.item()

                                if confidence > 0.5 and class_id in class_names:
                                    class_name = class_names[class_id]
                                    color = class_colors[class_id]
                                    counts[class_name] += 1

                                    # Draw bounding box
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                    cv2.putText(frame, f'{class_name}: {confidence:.2f}',
                                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5, color, 2)

                        # Add information overlay
                        self.add_info_overlay(frame, counts, inference_time)

                        # Update priority manager
                        self.priority_manager.update_lane_data(self.lane_id, counts)
                        self.priority_manager.determine_priority_lane()

                        # Store processed frame
                        with self.lock:
                            self.processed_frame = frame

                        # Update FPS
                        self.fps_counter += 1
                        if time.time() - self.fps_start_time >= 1.0:
                            self.current_fps = self.fps_counter / (time.time() - self.fps_start_time)
                            self.fps_counter = 0
                            self.fps_start_time = time.time()

            except Exception as e:
                print(f"[{self.name}] Error: {e}")
                time.sleep(1)

    def add_info_overlay(self, frame, counts, inference_time):
        """Add information overlay to frame"""
        # Camera name and lane ID
        cv2.putText(frame, f"{self.name} (Lane {self.lane_id})", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Object counts with colors
        y_pos = 50
        for class_name, count in counts.items():
            color = class_colors.get([k for k, v in class_names.items() if v == class_name][0], (255, 255, 255))
            cv2.putText(frame, f'{class_name.title()}: {count}', (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 25

        # Performance metrics
        cv2.putText(frame, f'Inference: {inference_time:.1f}ms', (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f'FPS: {self.current_fps:.1f}', (10, y_pos + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Priority indicator
        if counts.get('emergency', 0) > 0:
            cv2.putText(frame, "🚨 EMERGENCY DETECTED", (10, y_pos + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif counts.get('pedestrian', 0) > 0:
            cv2.putText(frame, "🚶 PEDESTRIAN DETECTED", (10, y_pos + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        elif self.lane_id == self.priority_manager.current_priority_lane:
            cv2.putText(frame, "🟢 PRIORITY LANE", (10, y_pos + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def display_frame(self):
        """Display the processed frame"""
        with self.lock:
            if self.processed_frame is not None:
                cv2.imshow(self.name, self.processed_frame)
                cv2.moveWindow(self.name, self.window_pos[0], self.window_pos[1])
                return True
        return False

    def save_frame(self):
        """Save current frame"""
        with self.lock:
            if self.latest_frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{save_dir}/{self.name}_{timestamp}.jpg"
                cv2.imwrite(filename, self.latest_frame)
                print(f"[{self.name}] Image saved: {filename}")
                return True
        return False

    def start(self):
        """Start camera thread"""
        self.thread = threading.Thread(target=self.grab_and_process_frames, daemon=True)
        self.thread.start()
        print(f"[{self.name}] Started")

    def cleanup(self):
        """Cleanup resources"""
        try:
            cv2.destroyWindow(self.name)
            self.session.close()
        except:
            pass


# ==============================
# Main Application
# ==============================
class TrafficManagementSystem:
    def __init__(self):
        self.stop_event = threading.Event()
        self.detectors = []
        self.serial_conn = None
        self.priority_manager = None
        self.setup_serial()
        self.setup_priority_manager()
        self.setup_detectors()

    def setup_serial(self):
        """Initialize serial connection to Arduino"""
        try:
            self.serial_conn = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1)
            time.sleep(2)  # Wait for Arduino to initialize
            print(f"✅ Serial connection established on {SERIAL_PORT}")
        except Exception as e:
            print(f"❌ Failed to establish serial connection: {e}")
            print("⚠️ Traffic lights will not be controlled")

    def setup_priority_manager(self):
        """Initialize traffic priority manager"""
        self.priority_manager = TrafficPriorityManager(self.serial_conn)

    def setup_detectors(self):
        """Initialize all camera detectors"""
        for cam_config in CAMERAS:
            detector = CameraDetector(cam_config, self.stop_event, self.priority_manager)
            self.detectors.append(detector)

    def start_all(self):
        """Start all camera detectors"""
        print("\n🚦 Starting Traffic Management System...")
        for detector in self.detectors:
            detector.start()
            time.sleep(1)  # Stagger startup

        print(f"\n✅ All {len(self.detectors)} cameras started successfully!")
        print("🖥️ Camera windows will appear as cameras connect...")

    def display_loop(self):
        """Main display loop"""
        print("\n=== CONTROLS ===")
        print("- Press 'q' on any window to quit")
        print("- Press 's' on focused window to save frame")
        print("- System automatically manages traffic priorities")
        print("==================\n")

        while not self.stop_event.is_set():
            quit_requested = False

            # Display frames from all cameras
            for detector in self.detectors:
                detector.display_frame()

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                quit_requested = True
            elif key == ord('s'):
                # Save frames from all cameras
                for detector in self.detectors:
                    detector.save_frame()

            if quit_requested:
                print("\n🛑 Shutdown requested...")
                break

            time.sleep(0.03)  # ~30 FPS display rate

    def stop_all(self):
        """Stop all systems"""
        print("🛑 Stopping Traffic Management System...")
        self.stop_event.set()

        # Give threads time to stop
        time.sleep(2)

        # Cleanup all detectors
        for detector in self.detectors:
            detector.cleanup()

        # Close serial connection
        if self.serial_conn:
            self.serial_conn.close()

        cv2.destroyAllWindows()
        print("✅ System stopped successfully.")

    def run(self):
        """Run the complete system"""
        try:
            self.start_all()
            self.display_loop()
        except KeyboardInterrupt:
            print("\n⚠️ Keyboard interrupt received...")
        finally:
            self.stop_all()


# ==============================
# Entry Point
# ==============================
if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"🎮 CUDA Device: {torch.cuda.get_device_name()}")
        print(f"💾 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\n🚦 AI Traffic Management System")
    print("🎯 Features: Priority detection, Emergency response, Pedestrian safety")

    system = TrafficManagementSystem()
    system.run()
