import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import queue


def test_yolo_parallelism():
    """Test if YOLO model can handle parallel inference efficiently"""

    print("🧪 Testing YOLO Parallelism")
    print("=" * 50)

    # Load single model
    model = YOLO("best.pt")
    device = next(model.model.parameters()).device
    print(f"Model device: {device}")

    # Create test frames
    test_frames = []
    for i in range(4):
        frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        # Add some variation to make frames different
        cv2.putText(frame, f'Frame {i + 1}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        test_frames.append(frame)

    # Test 1: Sequential processing
    print("\n📊 Test 1: Sequential Processing")
    start_time = time.time()

    for i, frame in enumerate(test_frames):
        inference_start = time.time()
        results = model(frame, verbose=False)
        inference_time = time.time() - inference_start
        print(f"Frame {i + 1}: {inference_time * 1000:.1f}ms")

    sequential_total = time.time() - start_time
    print(f"Sequential total: {sequential_total * 1000:.1f}ms")

    # Test 2: Parallel processing (shared model)
    print("\n📊 Test 2: Parallel Processing (Shared Model)")

    results_queue = queue.Queue()
    timing_queue = queue.Queue()

    def worker_thread(frame_id, frame):
        thread_start = time.time()
        inference_start = time.time()
        results = model(frame, verbose=False)
        inference_time = time.time() - inference_start
        total_time = time.time() - thread_start

        timing_queue.put((frame_id, inference_time, total_time))
        results_queue.put((frame_id, results))

    # Start all threads simultaneously
    parallel_start = time.time()
    threads = []

    for i, frame in enumerate(test_frames):
        thread = threading.Thread(target=worker_thread, args=(i + 1, frame))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    parallel_total = time.time() - parallel_start

    # Collect results
    print("Thread results:")
    thread_times = []
    while not timing_queue.empty():
        frame_id, inference_time, total_time = timing_queue.get()
        print(f"Frame {frame_id}: Inference {inference_time * 1000:.1f}ms, Total {total_time * 1000:.1f}ms")
        thread_times.append(inference_time)

    print(f"Parallel total: {parallel_total * 1000:.1f}ms")

    # Analysis
    print("\n📈 Performance Analysis:")
    print(f"Sequential: {sequential_total * 1000:.1f}ms")
    print(f"Parallel:   {parallel_total * 1000:.1f}ms")

    speedup = sequential_total / parallel_total
    print(f"Speedup:    {speedup:.1f}x")

    if speedup > 2:
        print("✅ Excellent parallelism! GPU is handling multiple streams efficiently")
    elif speedup > 1.5:
        print("⚡ Good parallelism! Some benefit from parallel processing")
    elif speedup > 1.1:
        print("🤔 Limited parallelism. GPU might be saturated or thread overhead")
    else:
        print("❌ No parallelism benefit. Sequential might be better")

    # GPU utilization insight
    avg_parallel_inference = np.mean(thread_times) if thread_times else 0
    print(f"\nGPU Utilization Insight:")
    print(f"Average inference time (parallel): {avg_parallel_inference * 1000:.1f}ms")

    if device.type == 'cuda':
        print("💡 With RTX 3070:")
        if avg_parallel_inference < 0.03:
            print("   GPU is very efficient, can handle more streams")
        elif avg_parallel_inference < 0.05:
            print("   GPU utilization is good")
        else:
            print("   GPU might be reaching limits or check CUDA setup")


def test_memory_usage():
    """Test memory usage with shared model"""
    import psutil
    import os

    print("\n🧠 Memory Usage Test")
    print("=" * 50)

    process = psutil.Process(os.getpid())

    # Before loading model
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory before model: {mem_before:.1f}MB")

    # After loading model
    model = YOLO("best.pt")
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    model_memory = mem_after - mem_before
    print(f"Memory after model:  {mem_after:.1f}MB")
    print(f"Model memory usage:  {model_memory:.1f}MB")

    # Test with multiple frames
    test_frames = [np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8) for _ in range(4)]

    mem_before_inference = process.memory_info().rss / 1024 / 1024

    # Run parallel inference
    def worker(frame):
        results = model(frame, verbose=False)
        return results

    threads = []
    for frame in test_frames:
        thread = threading.Thread(target=worker, args=(frame,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    mem_after_inference = process.memory_info().rss / 1024 / 1024
    inference_memory = mem_after_inference - mem_before_inference

    print(f"Memory during parallel inference: {mem_after_inference:.1f}MB")
    print(f"Additional memory for 4 streams: {inference_memory:.1f}MB")
    print(f"Memory per stream: {inference_memory / 4:.1f}MB")


if __name__ == "__main__":
    test_yolo_parallelism()
    test_memory_usage()