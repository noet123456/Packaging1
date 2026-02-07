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
    "rtsp://admin:KXILGD@192.168.1.152:554/ch1/main"   # CAM 4
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

CONFIRMATION_TIME = 1.0  # Giảm xuống 1s cho nhanh
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

def draw_text_bg(img, text, pos, font_scale=0.6, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Hàm phụ trợ để vẽ chữ có nền giúp dễ đọc hơn"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, 1)
    x, y = pos
    cv2.rectangle(img, (x, y - text_h - 4), (x + text_w, y + 4), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, 1)

def process_logic(frame, model, cam_idx, state_dict, collected_items):
    # Lấy state hiện tại
    state = state_dict.get(cam_idx, {'start': 0, 'ok': False, 'last': False, 'frozen_frame': None})
    
    # [LOGIC FREEZE] Nếu cam này đã OK, trả về frame đã đóng băng, KHÔNG chạy AI nữa
    if state['ok'] and state['frozen_frame'] is not None:
        return state['frozen_frame']

    # --- BẮT ĐẦU XỬ LÝ AI ---
    h_orig, w_orig = frame.shape[:2]
    # imgsz=640 để test nhanh, bạn có thể đổi lại 1024 nếu máy mạnh
    results = model.predict(frame, conf=0.5, imgsz=1024, verbose=False, device=DEVICE)[0]

    annotated = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
    sw, sh = DISPLAY_W / w_orig, DISPLAY_H / h_orig
    
    current_found = []

    # Vẽ kết quả
    if results.obb is not None:
        for i in range(len(results.obb.cls)):
            cls_id = int(results.obb.cls[i])
            raw_name = model.names[cls_id]
            clean_name = NAME_MAPPING.get(raw_name, raw_name)
            poly = (results.obb.xyxyxyxy[i].cpu().numpy() * [sw, sh]).astype(int)

            if "Slot" in raw_name:
                cv2.polylines(annotated, [poly], True, (255, 255, 0), 2)
            else:
                current_found.append(clean_name)
                cv2.polylines(annotated, [poly], True, (0, 255, 0), 2)
                # Vẽ tên linh kiện gọn gàng hơn
                cv2.putText(annotated, clean_name, (poly[0][0], poly[0][1] - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Logic kiểm tra
    required = CAM_REQUIREMENTS.get(cam_idx, [])
    # Cách kiểm tra: Clone list yêu cầu, xóa dần khi tìm thấy
    temp_req = required.copy()
    missing = []
    
    # Logic đối chiếu thông minh (xử lý trường hợp cần 3 cái giống nhau)
    found_copy = current_found.copy()
    for req_item in temp_req:
        if req_item in found_copy:
            found_copy.remove(req_item) # Xóa để không đếm trùng
        else:
            missing.append(req_item)
            
    is_complete = (len(missing) == 0)
    curr_time = time.time()

    if is_complete:
        if not state['last']:
            state['start'] = curr_time
            state['last'] = True
        elif curr_time - state['start'] >= CONFIRMATION_TIME:
            # --- CHUYỂN TRẠNG THÁI SANG PASSED ---
            state['ok'] = True
            
            # Cập nhật danh sách linh kiện đã thu thập vào Set chung
            for item in required:
                collected_items.add(item)
                
            # Vẽ giao diện lần cuối trước khi đóng băng
            cv2.rectangle(annotated, (0, 0), (DISPLAY_W, DISPLAY_H), (0, 255, 0), 4) # Viền xanh toàn khung
            draw_text_bg(annotated, f"CAM {cam_idx + 1}: PASSED", (10, 30), 0.8, (255, 255, 255), (0, 150, 0))
            
            # Lưu frame này lại để dùng cho các vòng lặp sau
            state['frozen_frame'] = annotated
    else:
        state['last'] = False
        state['ok'] = False
        
        # Vẽ giao diện khi đang thiếu
        cv2.rectangle(annotated, (0, 0), (DISPLAY_W, 40), (0, 0, 0), -1)
        if is_complete: # Đang chờ đếm ngược
             cv2.putText(annotated, f"CAM {cam_idx+1}: VERIFYING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
             cv2.putText(annotated, f"CAM {cam_idx+1}: MISSING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
             
        if missing:
            text_miss = f"Need: {', '.join(missing)}"
            cv2.rectangle(annotated, (0, DISPLAY_H - 30), (DISPLAY_W, DISPLAY_H), (0, 0, 0), -1)
            cv2.putText(annotated, text_miss, (10, DISPLAY_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    state_dict[cam_idx] = state
    return annotated

def main():
    print(f"--- Đang nạp Model trên {DEVICE} ---")
    # Thay đường dẫn model của bạn vào đây
    path_t1 = r"D:\YOLOv11n\Packaging\weights\t1\weights\best.pt"
    path_t2 = r"D:\YOLOv11n\Packaging\weights\t2\weights3\best.pt"
    
    # Kiểm tra đường dẫn tồn tại không để tránh crash
    if not os.path.exists(path_t1) or not os.path.exists(path_t2):
        print("Lỗi: Không tìm thấy file model!")
        return

    model_t1 = YOLO(path_t1)
    model_t2 = YOLO(path_t2)

    print("--- Đang kết nối Camera ---")
    cams = [CameraStream(url, f"C{i + 1}") for i, url in enumerate(RTSP_URLS)]
    
    states = {}
    collected_items = set() # Set chứa các linh kiện đã pass

    cv2.namedWindow("HE THONG KIEM TRA PACKAGING", cv2.WINDOW_NORMAL)

    print("--- Bắt đầu kiểm tra. Nhấn 'q' để thoát, 'r' để Reset hệ thống ---")

    while True:
        frames = []
        
        # Kiểm tra xem toàn hệ thống đã OK hết chưa
        # Giả sử cần 4 cam (index 0,1,2,3). Nếu bạn dùng ít cam hơn thì sửa range
        all_passed = True if len(states) == 4 else False
        for i in range(4):
            if not states.get(i, {}).get('ok', False):
                all_passed = False
                break
        
        for i in range(4):
            ret, img = cams[i].get_frame()
            if ret:
                m = model_t1 if i < 2 else model_t2
                # Truyền biến collected_items vào để ghi nhận
                processed = process_logic(img, m, i, states, collected_items)
                frames.append(processed)
            else:
                blank = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(blank, f"CAM {i + 1} NO SIGNAL", (150, 240), 1, 2, (0, 0, 255), 2)
                frames.append(blank)
                all_passed = False # Mất tín hiệu coi như chưa pass

        # Ghép hình 2x2
        top_row = np.hstack((frames[0], frames[1]))
        bottom_row = np.hstack((frames[2], frames[3]))
        combined = np.vstack((top_row, bottom_row))

        # --- GIAO DIỆN TỔNG HỆ THỐNG ---
        if all_passed:
            # Tạo overlay mờ màu xanh lá
            overlay = combined.copy()
            cv2.rectangle(overlay, (0, 0), (1280, 960), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.3, combined, 0.7, 0, combined)
            
            # Viết chữ SYSTEM PASSED to ở giữa
            cv2.putText(combined, "SYSTEM PASSED", (340, 480), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 10)
            cv2.putText(combined, "Press 'r' to restart", (480, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("HE THONG KIEM TRA PACKAGING", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Logic trả về kết quả khi thoát
            final_status = "SYSTEM PASSED" if all_passed else "SYSTEM FAILED"
            print("\n" + "="*30)
            print(f"KẾT QUẢ CUỐI CÙNG: {final_status}")
            print(f"Các linh kiện đã ghi nhận: {collected_items}")
            print("="*30)
            break
        elif key == ord('r'):
            # Reset trạng thái để kiểm tra hộp mới
            print("--- RESET SYSTEM ---")
            states = {}
            collected_items.clear()

    for c in cams:
        c.running = False
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()