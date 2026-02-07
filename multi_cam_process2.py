import cv2
import numpy as np
import torch
from ultralytics import YOLO
import threading
import time
import os

# --- CẤU HÌNH HỆ THỐNG ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- CẤU HÌNH CAMERA ---
RTSP_URLS = [
    "rtsp://admin:CPSFLT@192.168.1.160:554/ch1/main",  # CAM 1
    "rtsp://admin:DVCLRQ@192.168.1.116:554/ch1/main",  # CAM 2
    "rtsp://admin:BWKUYM@192.168.1.144:554/ch1/main",  # CAM 3
    "rtsp://admin:KXILGD@192.168.1.152:554/ch1/main"  # CAM 4
]

# Mapping tên linh kiện
NAME_MAPPING = {
    'Component 1': 'mainboard', 'Component 2': 'J-Link JTAG',
    'Component 3': 'Cam_bien_cd1', 'Component 4': 'Cam_bien_cd2', 'Component 5': 'Cam_bien_cd3',
    'Component 6': 'tui_module', 'Component 7': 'day_cap_trang', 'Component 8': 'day_cap_den',
    'Component 9': 'cap_bet', 'Component 10': 'jumper_mau'
}

# Yêu cầu kiểm tra cho từng Camera (Index 0-3)
CAM_REQUIREMENTS = {
    0: ['Cam_bien_cd1', 'Cam_bien_cd2', 'Cam_bien_cd3'],
    1: ['mainboard', 'J-Link JTAG'],
    2: ['day_cap_den', 'cap_bet', 'jumper_mau'],
    3: ['day_cap_trang', 'tui_module']
}

CONFIRMATION_TIME = 1.5


class CameraStream:
    def __init__(self, url, name):
        self.url = url
        self.name = name
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self.status = False
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                        self.status = True
                else:
                    self.status = False
                    self.cap.open(self.url)
            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if self.status and self.frame is not None:
                return True, self.frame.copy()
            return False, None


def process_logic(frame, model, cam_idx, state_dict):
    # Bước quan quan trọng: Ép kích thước xử lý
    # imgsz=1024 giúp AI soi được linh kiện nhỏ ở khoảng cách xa
    h_orig, w_orig = frame.shape[:2]
    results = model.predict(frame, conf=0.7, imgsz=1024, verbose=False, device=DEVICE)[0]

    # Kích thước hiển thị chuẩn cho mỗi ô cam
    display_w, display_h = 640, 480
    annotated = cv2.resize(frame, (display_w, display_h))

    # Tỉ lệ để vẽ lại tọa độ AI lên khung hình 640x480
    sw, sh = display_w / w_orig, display_h / h_orig

    found_items = []

    # Xử lý OBB (Box xoay)
    if results.obb is not None:
        for i in range(len(results.obb.cls)):
            cls_id = int(results.obb.cls[i])
            raw_name = model.names[cls_id]
            clean_name = NAME_MAPPING.get(raw_name, raw_name)
            poly = (results.obb.xyxyxyxy[i].cpu().numpy() * [sw, sh]).astype(int)

            if "Slot" in raw_name:
                cv2.polylines(annotated, [poly], True, (255, 255, 0), 2)
            else:
                found_items.append(clean_name)
                cv2.polylines(annotated, [poly], True, (0, 255, 0), 2)
                cv2.putText(annotated, clean_name, (poly[0][0], poly[0][1] - 5), 0, 0.5, (0, 255, 0), 1)

    # Logic kiểm tra PASSED/MISSING
    required = CAM_REQUIREMENTS.get(cam_idx, [])
    missing = [item for item in required if item not in found_items]
    is_complete = len(missing) == 0

    curr_time = time.time()
    state = state_dict.get(cam_idx, {'start': 0, 'ok': False, 'last': False})

    if is_complete:
        if not state['last']:
            state['start'] = curr_time
            state['last'] = True
        elif curr_time - state['start'] >= CONFIRMATION_TIME:
            state['ok'] = True
    else:
        state['last'] = False
        state['ok'] = False

    state_dict[cam_idx] = state

    # Vẽ giao diện thông báo
    header_color = (0, 255, 0) if state['ok'] else (0, 255, 255) if is_complete else (0, 0, 255)
    status_txt = "PASSED" if state['ok'] else "WAITING..." if is_complete else "MISSING"

    cv2.rectangle(annotated, (0, 0), (640, 45), (0, 0, 0), -1)
    cv2.putText(annotated, f"CAM {cam_idx + 1}: {status_txt}", (10, 32), 0, 0.8, header_color, 2)

    if missing:
        cv2.rectangle(annotated, (0, 450), (640, 480), (0, 0, 0), -1)
        cv2.putText(annotated, f"Need: {', '.join(missing)}", (10, 470), 0, 0.5, (0, 0, 255), 1)

    return annotated


def main():
    print(f"--- Đang nạp Model trên {DEVICE} ---")
    model_t1 = YOLO(r"D:\Packaging\weightsmoi\t1\best.pt")
    model_t2 = YOLO(r"D:\Packaging\weightsmoi\t2\weights3\best.pt")

    print("--- Đang kết nối Camera ---")
    cams = [CameraStream(url, f"C{i + 1}") for i, url in enumerate(RTSP_URLS)]
    states = {}

    cv2.namedWindow("HE THONG KIEM TRA PACKAGING", cv2.WINDOW_NORMAL)

    while True:
        frames = []
        for i in range(4):
            ret, img = cams[i].get_frame()
            if ret:
                # Phân tầng model: Cam 0,1 dùng model_t1 | Cam 2,3 dùng model_t2
                m = model_t1 if i < 2 else model_t2
                processed = process_logic(img, m, i, states)
                frames.append(processed)
            else:
                blank = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(blank, f"CAM {i + 1} LOSS SIGNAL", (150, 240), 0, 0.8, (0, 0, 255), 2)
                frames.append(blank)

        # FIX LỖI CROP: Đảm bảo 4 frame cùng size (480, 640) trước khi ghép
        # Ghép 2x2
        top_row = np.hstack((frames[0], frames[1]))
        bottom_row = np.hstack((frames[2], frames[3]))
        combined = np.vstack((top_row, bottom_row))

        # Hiển thị kết quả cuối cùng (tỉ lệ 1280x960)
        cv2.imshow("HE THONG KIEM TRA PACKAGING", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for c in cams:
        c.running = False
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()