import cv2
import numpy as np
import torch
from ultralytics import YOLO
import threading
import time
import os

# --- CẤU HÌNH HỆ THỐNG ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")

# --- CẤU HÌNH CAMERA ---
RTSP_URLS = [
    "rtsp://admin:CPSFLT@192.168.1.160:554/ch1/main", # CAM 1
    "rtsp://admin:DVCLRQ@192.168.1.116:554/ch1/main", # CAM 2
    "rtsp://admin:BWKUYM@192.168.1.144:554/ch1/main", # CAM 3
    "rtsp://admin:KXILGD@192.168.1.152:554/ch1/main"  # CAM 4
]

# --- CẤU HÌNH MAPPING VÀ YÊU CẦU ---
NAME_MAPPING = {
    'Component 9': 'cap_bet',
    'Component 1': 'mainboard',
    'Component 2': 'J-Link JTAG',
    'Component 3': 'Cam_bien_cd1',
    'Component 4': 'Cam_bien_cd2',
    'Component 5': 'Cam_bien_cd3',
    'Component 6': 'tui_module',
    'Component 7': 'day_cap_trang',
    'Component 8': 'day_cap_den',
    'Component 10': 'jumper_mau',
    'Slot 1': 'Trong',
    'Slot 2': 'Trong',
    'Slot 3': 'Trong',
    'Slot 4': 'Trong',
    'Slot 5': 'Trong',
    'Slot 6': 'Trong',
    'Slot 7': 'Trong',
    'Slot 8': 'Trong',
    'Slot 9': 'Trong',
    'Slot 10': 'Trong'
}

CAM_REQUIREMENTS = {
    0: ['Cam_bien_cd1', 'Cam_bien_cd2', 'Cam_bien_cd3'], # Component 1-3
    1: ['mainboard', 'J-Link JTAG'],                   # Component 4-5
    2: ['day_cap_den', 'cap_bet', 'jumper_mau'],       # Component 6-8
    3: ['day_cap_trang', 'tui_module']                 # Component 9-10
}

CONFIRMATION_TIME = 1.5 

# --- CLASS CAMERA STREAM ---
class CameraStream:
    def __init__(self, url, cam_name="Camera"):
        self.url = url
        self.name = cam_name
        self.capture = None
        self.current_frame = None
        self.status = False
        self.is_running = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        if self.is_running: return
        print(f"[{self.name}] Connecting...")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if self.capture.isOpened():
            self.is_running = True
            self.thread = threading.Thread(target=self.update, args=(), daemon=True)
            self.thread.start()
            print(f"[{self.name}] Started.")
        else:
            print(f"[{self.name}] Connection Failed.")

    def update(self):
        while self.is_running and self.capture.isOpened():
            self.capture.grab() # Keep buffer clean
            success, frame = self.capture.retrieve()
            if success:
                with self.lock:
                    self.current_frame = frame
                    self.status = True
            else:
                self.status = False
            time.sleep(0.01) # Small sleep

    def get_frame(self):
        with self.lock:
            if self.status and self.current_frame is not None:
                return True, self.current_frame.copy()
            return False, None

    def stop(self):
        print(f"[{self.name}] Stopping...")
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.capture:
            self.capture.release()
        self.status = False

# --- HÀM XỬ LÝ (Processing Logic) ---
def process_frame(frame, model, cam_idx):
    # 1. Resize to 640x480 (Full Frame, No Crop)
    frame_process = cv2.resize(frame, (640, 480))
    
    # 2. Single-Stage Prediction
    # Use standard boxes
    # Removed iou=0.5 to match testmodel.py
    results = model.predict(frame_process, conf=0.6, verbose=False)[0]
    
    found_items = []
    
    # 3. Draw Results
    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        
        # DEBUG: Print count/details if needed
        # if len(boxes) > 0: print(f"Cam {cam_idx}: {len(boxes)} boxes")
        
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = box.astype(int)
            cls_id = int(classes[i])
            conf = float(confs[i])
            raw_name = model.names[cls_id]
            label = NAME_MAPPING.get(raw_name, raw_name)
            
            # Filter 'Trong' or low confidence if needed
            if label == 'Trong':
                color = (0, 0, 255) # Red for Empty
            else:
                color = (0, 255, 0) # Green for Object
                found_items.append(label)

            cv2.rectangle(frame_process, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame_process, f"{label} {conf:.2f}", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    # Check Requirements
    required = CAM_REQUIREMENTS.get(cam_idx, [])
    missing = [item for item in required if item not in found_items]
    is_complete = (len(missing) == 0)

    if not is_complete:
        miss_str = ", ".join(missing)
        cv2.putText(frame_process, f"Missing: {miss_str}", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(frame_process, "OK", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    return frame_process, is_complete, (len(found_items) > 0)

# --- MAIN CONTROLLER ---
def main():
    # 1. Load Models
    print("Loading Models...")
    # T1 for Level 1 (Cam 1 & 2), T2 for Level 2 (Cam 3 & 4)
    # Correct paths based on user confirmation
    PATH_MODEL_1_5 = r"D:\YOLOv11n\Packaging\weights\t1\weights\best.pt" 
    PATH_MODEL_6_10 = r"D:\YOLOv11n\Packaging\weights\t2\weights3\best.pt"
    
    try:
        model_group1 = YOLO(PATH_MODEL_1_5) # For Cam 0, 1
        model_group2 = YOLO(PATH_MODEL_6_10) # For Cam 2, 3
        print("Models loaded.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 2. Init Camera Objects & Start ALL
    cameras = [CameraStream(url, f"Cam {i+1}") for i, url in enumerate(RTSP_URLS)]
    for cam in cameras:
        cam.start()
    
    # State Management
    # cam_states: 0=Active/Waiting, 1=Passed
    cam_states = [0] * 4 
    
    # Timers for EACH camera for independent verification
    cam_timers = [None] * 4 
    
    # Store latest frames for grid display
    display_buffers = [np.zeros((480, 640, 3), np.uint8) for _ in range(4)]
    
    # Final Result
    SYSTEM_DONE = False
    SYSTEM_RESULT_TEXT = ""
    SYSTEM_RESULT_COLOR = (0, 255, 0)

    print("System Started. Parallel Detection Mode (No Crop).")
    cv2.namedWindow("Production Monitor", cv2.WINDOW_NORMAL)

    try:
        while True:
            # Check for System Completion
            if all(s == 1 for s in cam_states):
                if not SYSTEM_DONE:
                    SYSTEM_DONE = True
                    SYSTEM_RESULT_TEXT = "SYSTEM PASSED"
            
            # PARALLEL LOOP
            for i, cam in enumerate(cameras):
                # If camera is already passed, we just skip processing (keep frozen frame in buffer)
                if cam_states[i] == 1:
                    continue
                
                ret, frame = cam.get_frame()
                if ret:
                    # Select Model
                    # Cam 0, 1 -> model_group1 (T1)
                    # Cam 2, 3 -> model_group2 (T2)
                    model = model_group1 if i < 2 else model_group2
                    
                    # Process Frame (Full Frame Resize)
                    processed, is_complete, found_any = process_frame(frame, model, i)
                    
                    # Update Buffer
                    display_buffers[i] = processed
                    
                    # Verification Logic (Independent for each camera)
                    if is_complete:
                        if cam_timers[i] is None:
                            cam_timers[i] = time.time()
                        else:
                            elapsed = time.time() - cam_timers[i]
                            cv2.putText(processed, f"VERIFY: {elapsed:.1f}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            
                            if elapsed >= CONFIRMATION_TIME:
                                # PASSED
                                print(f"Cam {i+1} PASSED.")
                                cam_states[i] = 1 # Mark passed
                                
                                # Freeze Frame with PASSED label
                                cv2.putText(processed, "PASSED", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                                display_buffers[i] = processed # Lock this frame
                    else:
                        cam_timers[i] = None # Reset timer
                else:
                    # Signal Lost
                    display_buffers[i] = np.zeros((480, 640, 3), np.uint8)
                    cv2.putText(display_buffers[i], f"CAM {i+1} SIGNAL LOST", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            # --- BUILD 2x2 GRID ---
            row1 = np.hstack((display_buffers[0], display_buffers[1]))
            row2 = np.hstack((display_buffers[2], display_buffers[3]))
            grid = np.vstack((row1, row2))

            if SYSTEM_DONE:
                 # Overlay System Result
                h, w = grid.shape[:2]
                cv2.rectangle(grid, (0, h//2 - 100), (w, h//2 + 100), SYSTEM_RESULT_COLOR, -1)
                cv2.putText(grid, SYSTEM_RESULT_TEXT, (w//2 - 300, h//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)

            cv2.imshow("Production Monitor", grid)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'): # Reset
                cam_states = [0]*4
                cam_timers = [None]*4
                SYSTEM_DONE = False
                SYSTEM_RESULT_TEXT = ""
                print("System Reset.")
            
    finally:
        print("Cleaning up...")
        for cam in cameras:
            cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
