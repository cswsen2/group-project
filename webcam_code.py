import cv2
import numpy as np
from ultralytics import YOLO
import time
from threading import Thread
import queue


class CarDetectionSystem:
    def __init__(self, model_path, camera_index=0):
        """
        Initialize the car detection system

        Args:
            model_path (str): Path to your trained YOLO11s model (best.pt)
            camera_index (int): Webcam index (usually 0 for default camera)
        """
        self.model = YOLO(model_path)
        self.camera_index = camera_index
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.car_count = 0

    def initialize_camera(self):
        """
        Initialize the webcam
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"✗ Error: Could not open camera {self.camera_index}")
            return False

        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        print(f"✓ Camera {self.camera_index} initialized successfully")
        return True

    def get_frame_from_webcam(self):
        """
        Capture frame from webcam
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            print("Failed to get frame from webcam")
            return None

    def frame_capture_thread(self):
        """
        Thread function to continuously capture frames from webcam
        """
        while self.running:
            frame = self.get_frame_from_webcam()
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
            time.sleep(0.033)  # ~30 FPS

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

    def test_camera_connection(self):
        """
        Test webcam connection
        """
        print(f"Testing connection to camera {self.camera_index}...")

        if not self.initialize_camera():
            print("✗ Failed to initialize camera")
            print("\nTroubleshooting tips:")
            print("1. Make sure no other application is using the camera")
            print("2. Try different camera index (0, 1, 2, etc.)")
            print("3. Check if camera is properly connected")
            print("4. Restart the application")
            return False

        # Test capture
        ret, frame = self.cap.read()
        if ret and frame is not None:
            print("✓ Camera is working properly")
            return True
        else:
            print("✗ Failed to capture test frame")
            return False

    def run(self):
        """
        Main function to run the car detection system
        """
        # Test camera connection first
        if not self.test_camera_connection():
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
                    cv2.imshow('Car Detection - Webcam', annotated_frame)

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
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Car detection system stopped")


def main():
    # Configuration
    MODEL_PATH = "best.pt"  # Your model file is in the same directory
    CAMERA_INDEX = 0  # Usually 0 for default webcam, try 1, 2, etc. if 0 doesn't work

    # Create and run the detection system
    detector = CarDetectionSystem(MODEL_PATH, CAMERA_INDEX)
    detector.run()


if __name__ == "__main__":
    main()


# Additional utility functions

def list_available_cameras():
    """
    Utility function to find available camera indices
    """
    print("Scanning for available cameras...")
    available_cameras = []

    for i in range(5):  # Check first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera {i}: Available")
                available_cameras.append(i)
            else:
                print(f"✗ Camera {i}: Cannot capture")
            cap.release()
        else:
            print(f"✗ Camera {i}: Not available")

    if available_cameras:
        print(f"\nAvailable cameras: {available_cameras}")
        print(f"Use CAMERA_INDEX = {available_cameras[0]} in your code")
    else:
        print("\nNo cameras found!")

    return available_cameras

# Usage example:
# python car_detection.py
