import cv2
import time
import requests
import numpy as np
from threading import Thread
from multiprocessing import Process, Queue
from ultralytics import YOLO


# ----------- FRAME READER FUNCTIONS ------------
def webcam_reader(queue, camera_index=0, name="Webcam"):
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Only keep latest frame
        while not queue.empty():
            try:
                queue.get_nowait()
            except:
                break
        queue.put((frame, name))
        time.sleep(0.033)  # Limit frame rate


def esp32_cam_reader(queue, url, name="ESP32-CAM"):
    while True:
        try:
            img_resp = requests.get(url, timeout=1)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            frame = cv2.imdecode(img_arr, -1)

            while not queue.empty():
                try:
                    queue.get_nowait()
                except:
                    break

            queue.put((frame, name))
        except:
            continue
        time.sleep(0.1)


# ----------- COMBINE INPUT QUEUES -------------
def combine_queues(webcam_q, esp32_q, combined_q):
    while True:
        if not webcam_q.empty():
            frame, name = webcam_q.get()
            combined_q.put((frame, name))
        if not esp32_q.empty():
            frame, name = esp32_q.get()
            combined_q.put((frame, name))
        time.sleep(0.01)


# ------------- YOLO DETECTION ------------------
def yolo_detector(model_path, input_q, output_q):
    model = YOLO(model_path)

    while True:
        if not input_q.empty():
            frame, name = input_q.get()
            results = model(frame, conf=0.72)
            result = results[0]
            car_count = 0

            if result.boxes is not None:
                for box in result.boxes:
                    if box.conf.item() >= 0.72:
                        car_count += 1

            annotated_frame = result.plot()
            output_q.put((annotated_frame, name, car_count))
        time.sleep(0.01)


# ------------- DISPLAY OUTPUT ------------------
def display_output(output_q):
    while True:
        if not output_q.empty():
            frame, name, count = output_q.get()
            label = f"{name} | Cars: {count}"
            cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# ------------------- MAIN ----------------------
if __name__ == "__main__":
    webcam_q = Queue(maxsize=2)
    esp32_q = Queue(maxsize=2)
    combined_q = Queue(maxsize=5)
    output_q = Queue(maxsize=5)

    esp32_url = 'http://10.96.228.144/cam-hi.jpg'  # Change this to your ESP32-CAM URL
    model_path = "best.pt"  # Your YOLO trained model

    # Start camera processes
    Process(target=webcam_reader, args=(webcam_q, 0, "Webcam")).start()
    Process(target=esp32_cam_reader, args=(esp32_q, esp32_url, "ESP32-CAM")).start()

    # Combine queues
    Process(target=combine_queues, args=(webcam_q, esp32_q, combined_q)).start()

    # Start YOLO inference process
    Process(target=yolo_detector, args=(model_path, combined_q, output_q)).start()

    # Display output
    display_output(output_q)
