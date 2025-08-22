import cv2
import numpy as np
from ultralytics import YOLO
import requests
import time
import threading
import queue
from datetime import datetime
import os


class MultiESP32CameraSystem:
    def __init__(self):
        # Camera configurations
        self.cameras = {
            "Camera_1": {"ip": "192.168.1.100", "position": "North Gate"},
            "Camera_2": {"ip": "192.168.1.101", "position": "South Gate"},
            "Camera_3": {"ip": "192.168.1.102", "position": "Parking Area"},
            "Camera_4": {"ip": "192.168.1.103", "position": "Main Entrance"}
        }

        # YOLO model - load once, use for all cameras
        print("Loading YOLO model...")
        self.model = YOLO("best.pt")

        # Force GPU usage if available
        device = 'cuda' if self.model.device.type == 'cuda' else 'cpu'
        print(f"Using device: {device}")

        # Class information
        self.class_info = {
            0: ('car', (0, 255, 0)),
            1: ('non-car', (0, 0, 255)),
            2: ('pedestrian', (255, 0, 0))
        }

        # Frame queues for each camera
        self.frame_queues = {cam: queue.Queue(maxsize=3) for cam in self.cameras}
        self.result_queues = {cam: queue.Queue(maxsize=3) for cam in self.cameras}

        # Control flags
        self.running = True
        self.sessions = {}

        # Statistics
        self.stats = {cam: {"fps": 0, "frames": 0, "errors": 0} for cam in self.cameras}

        # Create sessions for each camera
        for cam_name in self.cameras:
            self.sessions[cam_name] = requests.Session()
            self.sessions[cam_name].headers.update({'Connection': 'keep-alive'})

    def capture_frames(self, camera_name):
        """Capture frames from a specific ESP32-CAM"""
        camera_info = self.cameras[camera_name]
        url = f"http://{camera_info['ip']}/cam-mid.jpg"
        session = self.sessions[camera_name]

        frame_count = 0
        start_time = time.time()

        print(f"🎥 Started capturing from {camera_name} ({camera_info['position']})")

        while self.running:
            try:
                response = session.get(url, timeout=2.0)
                response.raise_for_status()

                img_array = np.frombuffer(response.content, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if frame is not None:
                    # Add camera info to frame
                    frame_with_info = {
                        'frame': frame,
                        'camera': camera_name,
                        'position': camera_info['position'],
                        'timestamp': datetime.now()
                    }

                    # Put frame in queue (non-blocking)
                    try:
                        self.frame_queues[camera_name].put(frame_with_info, block=False)
                        frame_count += 1
                    except queue.Full:
                        # Remove old frame and add new one
                        try:
                            self.frame_queues[camera_name].get_nowait()
                            self.frame_queues[camera_name].put(frame_with_info, block=False)
                        except queue.Empty:
                            pass

                # Update FPS stats
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    self.stats[camera_name]["fps"] = frame_count / elapsed
                    self.stats[camera_name]["frames"] = frame_count

            except Exception as e:
                self.stats[camera_name]["errors"] += 1
                if self.stats[camera_name]["errors"] % 10 == 1:  # Print every 10th error
                    print(f"❌ {camera_name} error: {e}")
                time.sleep(0.1)  # Brief pause on error

            time.sleep(0.033)  # ~30 FPS max attempt rate

    def process_frames(self):
        """Process frames from all cameras using YOLO"""
        print("🧠 Started YOLO processing thread")

        while self.running:
            # Process frames from all cameras
            for camera_name in self.cameras:
                try:
                    # Get frame from queue
                    frame_data = self.frame_queues[camera_name].get(timeout=0.1)

                    # Run YOLO detection
                    results = self.model(frame_data['frame'], conf=0.5, verbose=False)

                    # Process detections
                    processed_frame = self.draw_detections(
                        frame_data['frame'].copy(),
                        results[0],
                        frame_data['camera'],
                        frame_data['position']
                    )

                    # Put processed frame in result queue
                    result_data = {
                        'frame': processed_frame,
                        'camera': frame_data['camera'],
                        'position': frame_data['position'],
                        'timestamp': frame_data['timestamp']
                    }

                    try:
                        self.result_queues[camera_name].put(result_data, block=False)
                    except queue.Full:
                        try:
                            self.result_queues[camera_name].get_nowait()
                            self.result_queues[camera_name].put(result_data, block=False)
                        except queue.Empty:
                            pass

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Processing error for {camera_name}: {e}")

    def draw_detections(self, frame, results, camera_name, position):
        """Draw YOLO detections on frame"""
        counts = [0, 0, 0]  # car, non-car, pedestrian

        # Process detections
        if results.boxes is not None:
            boxes = results.boxes
            for i in range(len(boxes)):
                confidence = boxes.conf[i].item()
                if confidence > 0.5:
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    class_id = int(boxes.cls[i].item())

                    if class_id in self.class_info:
                        class_name, color = self.class_info[class_id]
                        counts[class_id] += 1

                        # Draw detection
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f'{class_name}: {confidence:.2f}',
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, color, 1)

        # Add camera info overlay
        overlay_height = 120
        overlay = np.zeros((overlay_height, frame.shape[1], 3), dtype=np.uint8)
        overlay[:] = (0, 0, 0)  # Black background

        # Camera info
        cv2.putText(overlay, f'Camera: {camera_name}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(overlay, f'Location: {position}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Detection counts
        cv2.putText(overlay, f'Cars: {counts[0]}', (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(overlay, f'Non-cars: {counts[1]}', (120, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(overlay, f'Pedestrians: {counts[2]}', (250, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # FPS info
        fps = self.stats[camera_name]["fps"]
        cv2.putText(overlay, f'FPS: {fps:.1f}', (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(overlay, timestamp, (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Combine overlay with frame
        combined = np.vstack([overlay, frame])
        return combined

    def display_cameras(self):
        """Display all camera feeds in a grid"""
        print("🖥️  Started display thread")

        # Create windows
        window_names = []
        for camera_name in self.cameras:
            window_name = f"{camera_name} - {self.cameras[camera_name]['position']}"
            window_names.append(window_name)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 640, 480)

        # Position windows in grid
        positions = [(0, 0), (650, 0), (0, 500), (650, 500)]
        for i, window_name in enumerate(window_names):
            if i < len(positions):
                cv2.moveWindow(window_name, positions[i][0], positions[i][1])

        while self.running:
            display_count = 0

            # Display each camera
            for i, camera_name in enumerate(self.cameras):
                try:
                    result_data = self.result_queues[camera_name].get(timeout=0.1)
                    cv2.imshow(window_names[i], result_data['frame'])
                    display_count += 1
                except queue.Empty:
                    continue

            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
            elif key == ord('s'):
                self.save_all_frames()
            elif key == ord('p'):
                self.print_stats()

        cv2.destroyAllWindows()

    def save_all_frames(self):
        """Save current frame from all cameras"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"saved_frames_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)

        for camera_name in self.cameras:
            try:
                result_data = self.result_queues[camera_name].get_nowait()
                filename = f"{save_dir}/{camera_name}_{timestamp}.jpg"
                cv2.imwrite(filename, result_data['frame'])
                print(f"💾 Saved {filename}")
            except queue.Empty:
                print(f"No frame available for {camera_name}")

    def print_stats(self):
        """Print performance statistics"""
        print("\n📊 System Statistics:")
        print("=" * 50)
        for camera_name, stats in self.stats.items():
            position = self.cameras[camera_name]['position']
            print(f"{camera_name} ({position}):")
            print(f"  FPS: {stats['fps']:.1f} | Frames: {stats['frames']} | Errors: {stats['errors']}")

    def run(self):
        """Main function to run the multi-camera system"""
        print("🚀 Starting 4x ESP32-CAM Detection System")
        print("📱 Camera Configuration:")
        for name, info in self.cameras.items():
            print(f"   {name}: {info['ip']} ({info['position']})")
        print("\nControls:")
        print("   'q' - Quit")
        print("   's' - Save all current frames")
        print("   'p' - Print performance stats")
        print("=" * 50)

        # Start capture threads for each camera
        capture_threads = []
        for camera_name in self.cameras:
            thread = threading.Thread(target=self.capture_frames, args=(camera_name,))
            thread.daemon = True
            thread.start()
            capture_threads.append(thread)

        # Start processing thread
        process_thread = threading.Thread(target=self.process_frames)
        process_thread.daemon = True
        process_thread.start()

        # Start display thread (main thread)
        try:
            self.display_cameras()
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            print("🛑 Shutting down...")

            # Close sessions
            for session in self.sessions.values():
                session.close()

            print("✅ Multi-camera system stopped")


def main():
    # Create and run the multi-camera system
    system = MultiESP32CameraSystem()

    # Update these IP addresses to match your ESP32-CAMs
    print("⚠️  Don't forget to update camera IP addresses in the code!")

    system.run()


if __name__ == "__main__":
    main()