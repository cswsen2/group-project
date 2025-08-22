import cv2
import numpy as np
from ultralytics import YOLO
import requests
import time
import threading
from datetime import datetime
import os


class OptimizedGaming4CameraSystem:
    def __init__(self):
        # Camera configurations
        self.cameras = {
            "Camera_1": {"ip": "192.168.1.100", "position": "North Gate"},
            "Camera_2": {"ip": "192.168.1.101", "position": "South Gate"},
            "Camera_3": {"ip": "192.168.1.102", "position": "Parking Area"},
            "Camera_4": {"ip": "192.168.1.103", "position": "Main Entrance"}
        }

        # Single YOLO model - GPU will share efficiently
        print("Loading YOLO model for gaming laptop...")
        self.model = YOLO("best.pt")

        # Verify GPU usage
        device = next(self.model.model.parameters()).device
        print(f"✅ Model running on: {device}")

        if device.type != 'cuda':
            print("⚠️  Warning: Not using GPU. Install CUDA-enabled PyTorch for best performance")

        # Class information
        self.class_info = {
            0: ('car', (0, 255, 0)),
            1: ('non-car', (0, 0, 255)),
            2: ('pedestrian', (255, 0, 0))
        }

        # Control flag
        self.running = True

        # Performance tracking
        self.stats = {
            cam: {
                "fps": 0,
                "frames": 0,
                "processing_times": [],
                "network_times": [],
                "start_time": time.time()
            } for cam in self.cameras
        }

        # Optimized sessions per camera
        self.sessions = {}
        for cam_name in self.cameras:
            session = requests.Session()
            # Optimize for high-performance networking
            session.headers.update({
                'Connection': 'keep-alive',
                'Keep-Alive': 'timeout=5, max=100'
            })
            self.sessions[cam_name] = session

    def camera_worker(self, camera_name):
        """
        Simplified worker: Capture → Process → Display
        No queues needed for high-end hardware
        """
        camera_info = self.cameras[camera_name]
        url = f"http://{camera_info['ip']}/cam-mid.jpg"
        session = self.sessions[camera_name]

        # Create dedicated window
        window_name = f"{camera_name} - {camera_info['position']}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)

        print(f"🎥 Started {camera_name} ({camera_info['position']})")

        frame_count = 0

        while self.running:
            try:
                # Network timing
                net_start = time.time()
                response = session.get(url, timeout=2.0)
                response.raise_for_status()

                img_array = np.frombuffer(response.content, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                net_time = time.time() - net_start

                if frame is None:
                    continue

                # YOLO processing timing
                process_start = time.time()
                results = self.model(frame, conf=0.5, verbose=False)
                process_time = time.time() - process_start

                # Draw detections efficiently
                annotated_frame = self.draw_detections(
                    frame, results[0], camera_name, camera_info['position'],
                    net_time, process_time
                )

                # Display immediately (no queue delay)
                cv2.imshow(window_name, annotated_frame)

                # Update statistics
                frame_count += 1
                self.stats[camera_name]["frames"] = frame_count
                self.stats[camera_name]["processing_times"].append(process_time)
                self.stats[camera_name]["network_times"].append(net_time)

                # Calculate FPS every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - self.stats[camera_name]["start_time"]
                    self.stats[camera_name]["fps"] = frame_count / elapsed

                # Handle input (any camera can trigger quit)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break

            except Exception as e:
                print(f"❌ {camera_name} error: {e}")
                time.sleep(0.1)

        cv2.destroyWindow(window_name)
        session.close()
        print(f"🛑 {camera_name} stopped")

    def draw_detections(self, frame, results, camera_name, position, net_time, process_time):
        """Optimized detection drawing for gaming laptop"""
        counts = [0, 0, 0]  # car, non-car, pedestrian

        # Process detections
        if results.boxes is not None:
            for i, box in enumerate(results.boxes):
                confidence = box.conf.item()
                if confidence > 0.5:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    class_id = int(box.cls.item())

                    if class_id in self.class_info:
                        class_name, color = self.class_info[class_id]
                        counts[class_id] += 1

                        # Draw detection
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f'{class_name}: {confidence:.2f}',
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, color, 1)

        # Gaming laptop info overlay
        info_height = 140
        overlay = np.zeros((info_height, frame.shape[1], 3), dtype=np.uint8)

        # Camera identification
        cv2.putText(overlay, f'Camera: {camera_name}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(overlay, f'Location: {position}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Detection counts with colors
        cv2.putText(overlay, f'Cars: {counts[0]}', (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(overlay, f'Non-cars: {counts[1]}', (120, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(overlay, f'Pedestrians: {counts[2]}', (250, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Performance metrics (gaming laptop specific)
        fps = self.stats[camera_name]["fps"]
        cv2.putText(overlay, f'FPS: {fps:.1f}', (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(overlay, f'GPU: {process_time * 1000:.0f}ms', (120, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(overlay, f'Network: {net_time * 1000:.0f}ms', (220, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # Hardware utilization indicator
        if process_time < 0.05:  # Less than 50ms = GPU working well
            gpu_status = "GPU: Optimal"
            gpu_color = (0, 255, 0)
        elif process_time < 0.1:
            gpu_status = "GPU: Good"
            gpu_color = (0, 255, 255)
        else:
            gpu_status = "GPU: Check CUDA"
            gpu_color = (0, 0, 255)

        cv2.putText(overlay, gpu_status, (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, gpu_color, 1)

        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(overlay, timestamp, (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return np.vstack([overlay, frame])

    def print_performance_stats(self):
        """Print detailed performance statistics for gaming laptop"""
        print("\n📊 Gaming Laptop Performance Stats:")
        print("=" * 60)

        for camera_name, stats in self.stats.items():
            position = self.cameras[camera_name]['position']

            if stats['processing_times'] and stats['network_times']:
                avg_process = np.mean(stats['processing_times'][-30:]) * 1000  # Last 30 frames
                avg_network = np.mean(stats['network_times'][-30:]) * 1000

                print(f"{camera_name} ({position}):")
                print(f"  FPS: {stats['fps']:.1f} | Frames: {stats['frames']}")
                print(f"  GPU Processing: {avg_process:.0f}ms | Network: {avg_network:.0f}ms")

                # Performance analysis
                if avg_process < 30:
                    print(f"  ✅ GPU Performance: Excellent")
                elif avg_process < 60:
                    print(f"  ⚡ GPU Performance: Good")
                else:
                    print(f"  ⚠️  GPU Performance: Check CUDA installation")

    def run(self):
        """Run the optimized 4-camera system"""
        print("🚀 Starting Gaming Laptop 4-Camera System")
        print("🎮 Hardware: i7 13th Gen + RTX 3070")
        print("⚡ Optimized: No queues, direct GPU processing")
        print("📱 Cameras:")
        for name, info in self.cameras.items():
            print(f"   {name}: {info['ip']} ({info['position']})")

        print("\n🎮 Gaming Laptop Controls:")
        print("   'q' - Quit (from any camera window)")
        print("   Windows will arrange automatically")
        print("=" * 50)

        # Position windows for gaming laptop (wider screen assumed)
        positions = [(0, 0), (660, 0), (1320, 0), (0, 650)]

        # Start all camera threads
        threads = []
        for i, camera_name in enumerate(self.cameras):
            thread = threading.Thread(target=self.camera_worker, args=(camera_name,))
            thread.daemon = True
            thread.start()
            threads.append(thread)

            # Brief delay to stagger window creation
            time.sleep(0.5)

        try:
            # Monitor system performance
            while self.running:
                time.sleep(10)  # Print stats every 10 seconds
                self.print_performance_stats()

        except KeyboardInterrupt:
            self.running = False

        print("\n🛑 Shutting down gaming system...")

        # Wait for threads to finish
        for thread in threads:
            thread.join(timeout=2)

        print("✅ Gaming laptop 4-camera system stopped")


def main():
    system = OptimizedGaming4CameraSystem()

    print("⚠️  Update camera IP addresses in the code!")
    print("⚠️  Ensure CUDA-enabled PyTorch is installed for GPU acceleration")

    system.run()


if __name__ == "__main__":
    main()