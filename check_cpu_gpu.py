import torch
from ultralytics import YOLO
import cv2
import numpy as np
import time


def check_available_devices():
    """Check what devices are available for YOLO"""

    print("🔍 Device Availability Check")
    print("=" * 50)

    # Check CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        print(f"✅ CUDA GPU Available")
        print(f"   GPU Count: {torch.cuda.device_count()}")
        print(f"   Current GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
        cuda_available = True
    else:
        print("❌ CUDA GPU Not Available")
        cuda_available = False

    # Check MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon GPU) Available")
        mps_available = True
    else:
        print("❌ MPS Not Available")
        mps_available = False

    # CPU is always available
    print("✅ CPU Available")

    return cuda_available, mps_available


def test_yolo_device_usage():
    """Test which device YOLO actually uses"""

    print("\n🧪 YOLO Device Usage Test")
    print("=" * 50)

    # Load model
    model = YOLO("best.pt")  # Replace with your model path

    # Create a dummy image
    dummy_frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)

    # Test inference and time it
    print("Running inference test...")

    # Warmup runs
    for _ in range(3):
        results = model(dummy_frame, verbose=False)

    # Actual timing
    times = []
    for i in range(10):
        start = time.time()
        results = model(dummy_frame, verbose=False)
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)

    # Check which device the model is on
    device = next(model.model.parameters()).device
    print(f"🎯 Model is running on: {device}")
    print(f"⏱️  Average inference time: {avg_time * 1000:.1f}ms")

    # Performance interpretation
    if avg_time < 0.05:  # Less than 50ms
        print("🚀 Performance: Excellent (likely GPU)")
    elif avg_time < 0.1:  # Less than 100ms
        print("⚡ Performance: Good (GPU or fast CPU)")
    elif avg_time < 0.5:  # Less than 500ms
        print("🐌 Performance: Moderate (likely CPU)")
    else:
        print("🐢 Performance: Slow (CPU or old hardware)")


def force_device_test():
    """Test YOLO on different devices explicitly"""

    print("\n🔧 Force Device Test")
    print("=" * 50)

    devices_to_test = ['cpu']

    if torch.cuda.is_available():
        devices_to_test.append('cuda')

    if torch.backends.mps.is_available():
        devices_to_test.append('mps')

    dummy_frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)

    for device_name in devices_to_test:
        try:
            print(f"\nTesting on {device_name.upper()}:")

            # Load model on specific device
            model = YOLO("best.pt")

            # Move model to device (for PyTorch models)
            if hasattr(model.model, 'to'):
                model.model.to(device_name)

            # Time inference
            start = time.time()
            results = model(dummy_frame, device=device_name, verbose=False)
            inference_time = time.time() - start

            print(f"  ✅ {device_name.upper()}: {inference_time * 1000:.1f}ms")

        except Exception as e:
            print(f"  ❌ {device_name.upper()}: Failed - {e}")


def main():
    """Main function to run all tests"""

    cuda_available, mps_available = check_available_devices()

    try:
        test_yolo_device_usage()
        force_device_test()

        print("\n💡 Recommendations:")
        if cuda_available:
            print("   - Your system can use GPU acceleration")
            print("   - YOLO should automatically use CUDA")
        elif mps_available:
            print("   - Your Mac can use MPS acceleration")
            print("   - YOLO should automatically use MPS")
        else:
            print("   - Only CPU available")
            print("   - Consider smaller YOLO model (yolo11n.pt)")

        print("\n🔍 To force CPU usage:")
        print("   results = model(frame, device='cpu')")
        print("🔍 To force GPU usage:")
        print("   results = model(frame, device='cuda')  # or 'mps' for Mac")

    except Exception as e:
        print(f"❌ Error testing model: {e}")
        print("Make sure 'best.pt' exists in your directory")


if __name__ == "__main__":
    main()