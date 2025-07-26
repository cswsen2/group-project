import cv2
import numpy as np
from ultralytics import YOLO
import requests
import time
from threading import Thread
import queue


class CarDetectionSystem:
    def __init__(self, model_path, esp32_ip):
        """
        Initialize the car detection system

        Args:
            model_path (str): Path to your trained YOLO11s model (best.pt)
            esp32_ip (str): IP address of your ESP32-CAM
        """
        self.model = YOLO(model_path)
        self.esp32_ip = esp32_ip
        self.esp32_url = f"http://{esp32_ip}"
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.car_count = 0

    def get_frame_from_esp32(self):
        """
        Capture frame from ESP32-CAM
        """
        # Common ESP32-CAM endpoints to try
        endpoints = [
            "/cam-hi.jpg",  # CameraWebServer example
            "/cam-lo.jpg",  # CameraWebServer example
            "/cam-mid.jpg",  # CameraWebServer example
            "/capture",  # Some custom implementations
            "/jpg",  # Some implementations
            "/cam.jpg",  # Alternative naming
            "/image",  # Another common endpoint
            "/photo"  # Another possibility
        ]

        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.esp32_url}{endpoint}",
                                        timeout=5, stream=True)

                if response.status_code == 200:
                    img_array = np.frombuffer(response.content, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame is not None:
                        print(f"✓ Using endpoint: {endpoint}")
                        # Store the working endpoint for future use
                        self.working_endpoint = endpoint
                        return frame

            except requests.exceptions.RequestException:
                continue

        # If we have a working endpoint from before, use it
        if hasattr(self, 'working_endpoint'):
            try:
                response = requests.get(f"{self.esp32_url}{self.working_endpoint}",
                                        timeout=5, stream=True)
                if response.status_code == 200:
                    img_array = np.frombuffer(response.content, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    return frame
            except:
                pass

        print("Failed to get frame from any endpoint")
        return None

    def frame_capture_thread(self):
        """
        Thread function to continuously capture frames from ESP32-CAM
        """
        while self.running:
            frame = self.get_frame_from_esp32()
            if frame is not None:
                # Add frame to queue, remove old frame if queue is full
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    try:
                        self.frame_queue.get_nowait()  # Remove old frame
                        self.frame_queue.put(frame)  # Add new frame
                    except queue.Empty:
                        pass
            time.sleep(0.1)  # Adjust based on desired frame rate

    def detect_cars(self, frame):
        """
        Detect cars in the frame using YOLO11s

        Args:
            frame: Input image frame

        Returns:
            annotated_frame: Frame with bounding boxes
            car_count: Number of cars detected
        """
        # Run YOLO inference with higher confidence threshold
        results = self.model(frame, conf=0.72)  # Only detections above 72% confidence

        # Get the first result (since we're processing one image)
        result = results[0]

        # Count cars with confidence >= 0.72
        car_count = 0
        if result.boxes is not None:
            for box in result.boxes:
                if box.conf.item() >= 0.72:  # Additional check for confidence
                    car_count += 1

        # Draw bounding boxes and labels (only for high confidence detections)
        annotated_frame = result.plot()

        # Add car count text on the frame
        cv2.putText(annotated_frame, f'Cars Detected (>72%): {car_count}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_frame, timestamp,
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated_frame, car_count

    def print_car_count(self, car_count):
        """
        Simply print the car count
        """
        print(f"Cars detected: {car_count}")

    def test_esp32_connection(self):
        """
        Test connection to ESP32-CAM and find working endpoint
        """
        print(f"Testing connection to ESP32-CAM at {self.esp32_ip}...")

        # Common ESP32-CAM endpoints to try
        endpoints = [
            "/cam-hi.jpg",  # CameraWebServer example
            "/cam-lo.jpg",  # CameraWebServer example
            "/cam-mid.jpg",  # CameraWebServer example
            "/capture",  # Some custom implementations
            "/jpg",  # Some implementations
            "/cam.jpg",  # Alternative naming
            "/image",  # Another common endpoint
            "/photo"  # Another possibility
        ]

        for endpoint in endpoints:
            try:
                print(f"Trying endpoint: {endpoint}")
                response = requests.get(f"{self.esp32_url}{endpoint}", timeout=5)

                if response.status_code == 200:
                    # Check if it's actually an image
                    img_array = np.frombuffer(response.content, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    if frame is not None:
                        print(f"✓ Successfully connected using endpoint: {endpoint}")
                        self.working_endpoint = endpoint
                        return True

            except requests.exceptions.RequestException as e:
                print(f"  Failed: {e}")
                continue

        print("✗ Could not find any working endpoint")
        print("\nCommon ESP32-CAM setup issues:")
        print("1. Make sure ESP32-CAM is powered on and connected to WiFi")
        print("2. Check if you can access the web interface in browser:")
        print(f"   http://{self.esp32_ip}")
        print("3. Verify the IP address is correct")
        print("4. Make sure you're on the same network as ESP32-CAM")
        return False

    def run(self):
        """
        Main function to run the car detection system
        """
        # Test ESP32-CAM connection first
        if not self.test_esp32_connection():
            return

        print("Starting car detection system...")
        print("Press 'q' to quit, 's' to save current frame")

        self.running = True

        # Start frame capture thread
        capture_thread = Thread(target=self.frame_capture_thread)
        capture_thread.daemon = True
        capture_thread.start()

        # Main processing loop
        while self.running:
            try:
                # Get frame from queue
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()

                    # Detect cars
                    annotated_frame, self.car_count = self.detect_cars(frame)

                    # Print car count
                    self.print_car_count(self.car_count)

                    # Display the frame
                    cv2.imshow('Car Detection - Traffic Control System', annotated_frame)

                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save current frame
                        filename = f"detection_{int(time.time())}.jpg"
                        cv2.imwrite(filename, annotated_frame)
                        print(f"Frame saved as {filename}")

            except KeyboardInterrupt:
                break

        # Cleanup
        self.running = False
        cv2.destroyAllWindows()
        print("Car detection system stopped")


def main():
    # Configuration
    MODEL_PATH = "best.pt"  # Your model file is in the same directory
    ESP32_IP = "10.61.31.144"  # Your ESP32-CAM IP address

    # You can find your ESP32-CAM IP by:
    # 1. Check your router's connected devices
    # 2. Use ESP32-CAM serial monitor
    # 3. Use network scanner apps

    # Create and run the detection system
    detector = CarDetectionSystem(MODEL_PATH, ESP32_IP)
    detector.run()


if __name__ == "__main__":
    main()


# Additional utility functions for ESP32-CAM setup

def find_esp32_ip():
    """
    Utility function to help find ESP32-CAM IP address
    """
    import socket
    import subprocess

    # Get your computer's IP to determine network range
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    network = '.'.join(local_ip.split('.')[:-1])

    print(f"Scanning network {network}.1-254 for ESP32-CAM...")
    print("This may take a few minutes...")

    # Common ESP32-CAM endpoints to test
    endpoints = ['/cam-hi.jpg', '/cam-lo.jpg', '/capture']

    for i in range(1, 255):
        ip = f"{network}.{i}"
        try:
            for endpoint in endpoints:
                response = requests.get(f"http://{ip}{endpoint}", timeout=1)
                if response.status_code == 200:
                    print(f"Found ESP32-CAM at {ip}")
                    return ip
        except:
            continue

    print("ESP32-CAM not found. Make sure it's connected to the same network.")
    return None

# Usage example:
# python car_detection.py