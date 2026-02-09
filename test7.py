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
    "rtsp://admin:CPSFLT@192.168.1.160:554/ch1/main",
    "rtsp://admin:DVCLRQ@192.168.1.116:554/ch1/main",
    "rtsp://admin:BWKUYM@192.168.1.144:554/ch1/main",
    "rtsp://admin:KXILGD@192.168.1.152:554/ch1/main"
]

NAME_MAPPING = {
    'Component 1': 'mainboard', 'Component 2': 'J-Link JTAG',
    'Component 3': 'Cam_bien_cd', 'Component 4': 'Cam_bien_cd', 'Component 5': 'Cam_bien_cd',
    'Component 6': 'tui_module', 'Component 7': 'day_cap_trang', 'Component 8': 'day_cap_den',
    'Component 9': 'cap_bet', 'Component 10': 'jumper_mau'
}

CAM_REQUIREMENTS = {
    0: ['Cam_bien_cd', 'Cam_bien_cd', 'Cam_bien_cd'],
    1: ['mainboard', 'J-Link JTAG'],
    2: ['day_cap_den', 'cap_bet', 'jumper_mau'],
    3: ['day_cap_trang', 'tui_module']
}

MISSING_FINAL = []

CONFIRMATION_TIME = 1.0
DISPLAY_W, DISPLAY_H = 640, 480

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

# Hàm vẽ màn hình kết quả cuối cùng
def show_final_result(image, passed, missing_cams):
    overlay = image.copy()
    h, w = image.shape[:2]
    
    if passed:
        color = (0, 255, 0) # Xanh lá
        text = "SYSTEM PASSED"
        sub_text = "All components verified."
    else:
        color = (0, 0, 255) # Đỏ
        text = "SYSTEM FAILED"
        # Liệt kê các camera chưa đạt
        sub_text = f"Failed at Cam: {', '.join(map(str, missing_cams))}"

    # Làm tối màn hình
    cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
    cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)

    # Tính toán vị trí text để căn giữa
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 3, 5)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (h + text_size[1]) // 2
    
    cv2.putText(image, text, (text_x, text_y), font, 3, (255, 255, 255), 5)
    
    sub_size = cv2.getTextSize(sub_text, font, 1, 2)[0]
    sub_x = (w - sub_size[0]) // 2
    cv2.putText(image, sub_text, (sub_x, text_y + 60), font, 1, (255, 255, 255), 2)
    
    return image

def process_logic(frame, model, cam_idx, state_dict, collected_items):  #104 - 162
    state = state_dict.get(cam_idx, {'start': 0, 'ok': False, 'last': False, 'frozen_frame': None})
    
    if state['ok'] and state['frozen_frame'] is not None:
        return state['frozen_frame']

    results = model.predict(frame, conf=0.5, imgsz=640, verbose=False, device=DEVICE)[0] # imgsz=640 cho nhanh
    h_orig, w_orig = frame.shape[:2]
    annotated = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
    sw, sh = DISPLAY_W / w_orig, DISPLAY_H / h_orig
    
    current_found = []

    if results.obb is not None:
        for i in range(len(results.obb.cls)):
            cls_id = int(results.obb.cls[i])
            raw_name = model.names[cls_id]
            clean_name = NAME_MAPPING.get(raw_name, raw_name)
            poly = (results.obb.xyxyxyxy[i].cpu().numpy() * [sw, sh]).astype(int)

            if "Slot" in raw_name:
                cv2.polylines(annotated, [poly], True, (255, 255, 0), 2)
                cv2.putText(annotated, clean_name, (poly[0][0], poly[0][1] - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            else:
                current_found.append(clean_name)
                cv2.polylines(annotated, [poly], True, (0, 255, 0), 2)
                cv2.putText(annotated, clean_name, (poly[0][0], poly[0][1] - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    required = CAM_REQUIREMENTS.get(cam_idx, [])
    temp_req = required.copy()
    missing = []
    found_copy = current_found.copy()
    global MISSING_FINAL
    MISSING_FINAL.append(temp_req)
    
    for req_item in temp_req:
        if req_item in found_copy:
            found_copy.remove(req_item)
            MISSING_FINAL.remove(req_item)
        else:
            missing.append(req_item)
            
    is_complete = (len(missing) == 0)
    curr_time = time.time()

    if is_complete:
        if not state['last']:
            state['start'] = curr_time
            state['last'] = True
        elif curr_time - state['start'] >= CONFIRMATION_TIME:
            state['ok'] = True
            for item in required:
                collected_items.add(item)
            
            cv2.rectangle(annotated, (0, 0), (DISPLAY_W, DISPLAY_H), (0, 255, 0), 4)
            cv2.putText(annotated, f"CAM {cam_idx + 1}: PASSED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            state['frozen_frame'] = annotated
    else:
        state['last'] = False
        state['ok'] = False
        # Hiển thị trạng thái
        cv2.rectangle(annotated, (0, 0), (DISPLAY_W, 40), (0, 0, 0), -1)
        status_text = "VERIFYING..." if is_complete else "MISSING"
        color_status = (0, 255, 255) if is_complete else (0, 0, 255)
        cv2.putText(annotated, f"CAM {cam_idx+1}: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_status, 2)
        
        if missing:
             cv2.putText(annotated, f"Need: {', '.join(missing)}", (10, DISPLAY_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    state_dict[cam_idx] = state
    return annotated

def main():
    print(f"--- Loading Model on {DEVICE} ---")
    path_t1 = r"D:\YOLOv11n\Packaging\weights\t1\weights\stage1.pt"
    path_t2 = r"D:\YOLOv11n\Packaging\weights\t2\weights3\stage2.pt"
    
    model_t1 = YOLO(path_t1)
    model_t2 = YOLO(path_t2)

    cams = [CameraStream(url, f"C{i + 1}") for i, url in enumerate(RTSP_URLS)]
    states = {}
    collected_items = set()

    cv2.namedWindow("HE THONG KIEM TRA PACKAGING", cv2.WINDOW_NORMAL)
    print("--- Running... Press 'q' to FINISH CHECK, 'r' to RESET ---")

    running = True
    while running:
        frames = []
        missing_cams_idx = []
        
        # Check overall status
        all_passed = True
        for i in range(4):
            if not states.get(i, {}).get('ok', False):
                all_passed = False
                missing_cams_idx.append(i + 1)

        for i in range(4):
            ret, img = cams[i].get_frame()
            if ret:
                m = model_t1 if i < 2 else model_t2
                processed = process_logic(img, m, i, states, collected_items)
                frames.append(processed)
            else:
                blank = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(blank, f"NO SIGNAL", (200, 240), 1, 2, (0, 0, 255), 2)
                frames.append(blank)

        # Ghép hình
        top_row = np.hstack((frames[0], frames[1]))
        bottom_row = np.hstack((frames[2], frames[3]))
        combined = np.vstack((top_row, bottom_row))

        # Nếu đã PASS hết -> Tự động hiện SYSTEM PASSED (Overlay xanh)
        if all_passed:
            # Vẽ overlay trực tiếp lên frame hiện tại nhưng chưa dừng hẳn
            overlay = combined.copy()
            cv2.rectangle(overlay, (0, 0), (1280, 960), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.3, combined, 0.7, 0, combined)
            cv2.putText(combined, "SYSTEM PASSED", (340, 480), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
            cv2.putText(combined, "Press 'r' to New Box or 'q' to Exit", (380, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
        # Chưa PASS hiện SYSTEM FAILED (Overlay đỏ)
            overlay = combined.copy()
            cv2.rectangle(overlay, (0, 0), (1280, 960), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, combined, 0.7, 0, combined)
            cv2.putText(combined, "SYSTEM FAILED", (340, 480), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
            cv2.putText(combined, "Press 'r' to New Box or 'q' to Exit", (380, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) 
            cv2.putText(combined, f"Missing: {', '.join(MISSING_FINAL)}", (380, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  

        cv2.imshow("HE THONG KIEM TRA PACKAGING", combined)

        key = cv2.waitKey(1) & 0xFF
        
        # --- XỬ LÝ PHÍM BẤM ---
        if key == ord('q'):
            # Khi bấm Q, ta sẽ hiện kết quả cuối cùng lên màn hình trước khi thoát
            final_img = show_final_result(combined, all_passed, missing_cams_idx)
            cv2.imshow("HE THONG KIEM TRA PACKAGING", final_img)
            
            # Giữ hình ảnh kết quả trong 3 giây để người dùng nhìn thấy
            cv2.waitKey(3000) 
            running = False # Thoát vòng lặp
            
        elif key == ord('r'):
            # Reset
            states = {}
            collected_items.clear()
            print("--- RESET SYSTEM ---")

    # In kết quả ra console sau khi cửa sổ đã đóng (để log lại nếu cần)
    print("\n" + "="*30)
    print(f"KẾT QUẢ: {'PASSED' if all_passed else 'FAILED'}")
    print(f"Linh kiện đã tìm thấy: {collected_items}")
    print("="*30)

    for c in cams:
        c.running = False
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
