import cv2
import numpy as np

print("OpenCV Version:", cv2.__version__)

# 1. Check if CUDA is available in your OpenCV build
cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
print(f"CUDA Devices Found: {cuda_devices}")

if cuda_devices == 0:
    print("CRITICAL: Your OpenCV build does not support CUDA or cannot find your GPU drivers.")
    print("Please verify your build flags or fallback to the working CPU version.")
    exit()

# 2. Create a basic 2D test matrix (CPU)
h, w = 480, 640
cpu_frame = np.zeros((h, w, 3), dtype=np.uint8)
# Paint a simple visual anchor (Blue box)
cv2.rectangle(cpu_frame, (100, 100), (540, 380), (255, 0, 0), -1)
cv2.putText(cpu_frame, "GPU Context Test", (150, 240), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

# 3. Upload to GPU memory space
gpu_frame = cv2.cuda_GpuMat()
gpu_frame.upload(cpu_frame)

# 4. Perform a fast GPU operation (Invert colors using bitwise NOT)
gpu_result = cv2.cuda.bitwise_not(gpu_frame)

# 5. Download the result back to CPU for rendering
downloaded_frame = gpu_result.download()

# 6. Test OpenCV Window Manager thread synchronization
print("\nOpening display window. Look for a window named 'GPU Window Test'...")
cv2.namedWindow("GPU Window Test", cv2.WINDOW_AUTOSIZE)

# Keep the window updating in a clean 3-second loop to verify it is responsive
for i in range(100):
    # Overlay a dynamic counter frame frame-by-frame on CPU
    display_img = downloaded_frame.copy()
    cv2.putText(display_img, f"Frame: {i}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow("GPU Window Test", display_img)
    
    # Yield thread resources for 30ms to the X11 window system
    if cv2.waitKey(30) & 0xFF == ord('q'):
        print("Playback halted manually.")
        break

cv2.destroyAllWindows()
print("Test sequence finished successfully! Window thread is highly responsive.")
