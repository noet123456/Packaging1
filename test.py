import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os

# --- THIẾT LẬP HỆ THỐNG ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. TỪ ĐIỂN ÁNH XẠ (MAPPING)
# Chuyển tên gốc từ model sang tên tiếng Việt hoặc tên dễ hiểu
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
    'o trong': 'Trong'
}

# 2. CẤU HÌNH YÊU CẦU THEO TỪNG CAMERA
CAM_REQUIREMENTS = {
    0: ['Cam_bien_cd1', 'Cam_bien_cd2', 'Cam_bien_cd3'],  # Cam 1
    1: ['mainboard', 'J-Link JTAG'],  # Cam 2
    2: ['day_cap_den', 'cap_bet', 'jumper_mau'],  # Cam 3
    3: ['day_cap_trang', 'tui_module']  # Cam 4
}


# --- CÁC HÀM XỬ LÝ ---

def center_crop(frame, crop_width=640, crop_height=480):
    """Cắt vùng ảnh trung tâm của Camera"""
    h, w = frame.shape[:2]
    start_x = max(0, w // 2 - crop_width // 2)
    start_y = max(0, h // 2 - crop_height // 2)
    return frame[start_y:start_y + crop_height, start_x:start_x + crop_width]


def extract_detections(results):
    """Trích xuất tọa độ OBB từ kết quả YOLO"""
    detections = []
    r = results[0]
    if r.obb is not None:
        boxes_xywhr = r.obb.xywhr.cpu().numpy()
        cls_ids = r.obb.cls.cpu().numpy()
        xyxyxyxy = r.obb.xyxyxyxy.cpu().numpy()
        for i in range(len(cls_ids)):
            detections.append({
                'xywhr': boxes_xywhr[i],
                'poly': xyxyxyxy[i].astype(np.int32),
                'cls_id': int(cls_ids[i])
            })
    return detections


def predict_component_yolo(model, frame, poly):
    """Nhận diện linh kiện trong vùng crop bằng model Detection"""
    x, y, w, h = cv2.boundingRect(poly)
    img_h, img_w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x), max(0, y), min(img_w, x + w), min(img_h, y + h)
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return NAME_MAPPING['o trong'], 0.0

    results = model.predict(crop, conf=0.3, verbose=False)[0]

    if len(results.boxes) > 0:
        best_box = results.boxes[0]
        idx = int(best_box.cls[0])
        conf = float(best_box.conf[0])
        # Lấy tên từ model.names đã được nạp từ file .pth
        raw_name = model.names[idx]
        return NAME_MAPPING.get(raw_name, raw_name), conf

    return NAME_MAPPING['o trong'], 0.0


# --- LUỒNG CHÍNH ---

if __name__ == "__main__":
    print("Đang khởi tạo hệ thống giám sát...")

    # 1. Load YOLO tìm slot (Dùng file .pt chuẩn)
    m_slot_t1 = YOLO(r"D:\YOLOv11n\test\weights\t1\stage1.pt")
    m_slot_t2 = YOLO(r"D:\YOLOv11n\test\weights\t2\stage2.pt")

    # 2. Load YOLO nhận diện linh kiện (Xử lý file .pth đặc biệt)
    m_comp = YOLO("yolo11n.pt")
    checkpoint = torch.load(r"D:\YOLOv11n\test\weights\linhkien.pth", map_location=device)

    if 'stage1' in checkpoint:
        # Nạp trọng số với strict=False để tránh lỗi size mismatch
        # --- SỬA ĐỔI: NẠP MÔ HÌNH LINH KIỆN ---
        # Sử dụng Stage1.pt làm khung vì nó đã chạy ổn định trên máy bạn
        m_comp = YOLO(r"D:\YOLOv11n\test\weights\t1\stage1.pt")

        checkpoint = torch.load(r"D:\YOLOv11n\test\weights\linhkien.pth", map_location=device)

        # Kiểm tra và nạp trọng số
        if 'stage1' in checkpoint:
            # Lấy state_dict từ file .pth
            state_dict = checkpoint['stage1']

            # ÉP BUỘC NẠP: Chúng ta nạp vào model nội bộ của đối tượng YOLO
            # Lưu ý: model.load_state_dict của PyTorch vẫn sẽ lỗi nếu cấu trúc sai,
            # nên ta cần đảm bảo m_comp khởi tạo đúng bản Nano.
            try:
                m_comp.model.load_state_dict(state_dict, strict=True)
                print("Nạp trọng số linh kiện thành công.")
            except RuntimeError:
                print("Phát hiện lệch kiến trúc! Đang thử nạp với chế độ tương thích...")
                m_comp.model.load_state_dict(state_dict, strict=False)

            # Cập nhật danh sách tên Class
            custom_names = checkpoint.get('model_1_names', {})
            if custom_names:
                m_comp.model.names = custom_names
        print("Đã nạp thành công trọng số linh kiện.")

    m_comp.to(device)

    # 3. Cấu hình Camera
    rtsp_urls = [
        "rtsp://admin:CPSFLT@192.168.1.160:554/ch1/main",
        "rtsp://admin:DVCLRQ@192.168.1.116:554/ch1/main",
        "rtsp://admin:BWKUYM@192.168.1.144:554/ch1/main",
        "rtsp://admin:KXILGD@192.168.1.152:554/ch1/main"
    ]
    caps = [cv2.VideoCapture(url) for url in rtsp_urls]
    W_CROP, H_CROP = 640, 480  # Bạn có thể tăng lên 1280, 720 nếu muốn nhìn rộng hơn

    # Tạo cửa sổ hiển thị có khả năng kéo giãn
    cv2.namedWindow("Production Monitor", cv2.WINDOW_NORMAL)

    while True:
        processed_frames = []
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                err = np.zeros((H_CROP, W_CROP, 3), np.uint8)
                cv2.putText(err, f"CAM {i + 1} LOST", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                processed_frames.append(err)
                continue

            frame_c = center_crop(frame, W_CROP, H_CROP)
            m_slot = m_slot_t1 if i < 2 else m_slot_t2

            # Dự đoán vị trí ô (Slot)
            res_slot = m_slot.predict(frame_c, conf=0.5, verbose=False)
            slots = extract_detections(res_slot)
            found_items = []

            # Kiểm tra từng ô để xem linh kiện gì
            for s in slots:
                name, conf = predict_component_yolo(m_comp, frame_c, s['poly'])
                if name != NAME_MAPPING['o trong']:
                    found_items.append(name)

                # Vẽ khung linh kiện
                color = (0, 255, 0) if name != NAME_MAPPING['o trong'] else (0, 0, 255)
                cv2.polylines(frame_c, [s['poly']], True, color, 2)
                cv2.putText(frame_c, f"{name}", (s['poly'][0][0], s['poly'][0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Kiểm tra Logic Đủ/Thiếu
            req = CAM_REQUIREMENTS.get(i, [])
            miss = [x for x in req if x not in found_items]
            is_ok = len(miss) == 0

            # Vẽ bảng trạng thái (Status Bar)
            cv2.rectangle(frame_c, (0, 0), (320, 65), (0, 0, 0), -1)
            st_text = "OK" if is_ok else "MISSING"
            st_color = (0, 255, 0) if is_ok else (0, 0, 255)
            cv2.putText(frame_c, f"CAM {i + 1}: {st_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, st_color, 2)
            if not is_ok:
                cv2.putText(frame_c, f"Need: {', '.join(miss)}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            processed_frames.append(frame_c)

        # Hiển thị Grid 2x2
        if len(processed_frames) == 4:
            top_row = np.hstack((processed_frames[0], processed_frames[1]))
            bottom_row = np.hstack((processed_frames[2], processed_frames[3]))
            full_grid = np.vstack((top_row, bottom_row))

            # --- PHÓNG ĐẠI MÀN HÌNH (Chỉnh scale ở đây) ---
            scale = 1.4  # Tăng lên 1.4 lần cho rộng hơn
            w_large = int(full_grid.shape[1] * scale)
            h_large = int(full_grid.shape[0] * scale)
            grid_final = cv2.resize(full_grid, (w_large, h_large), interpolation=cv2.INTER_LINEAR)

            cv2.imshow("Production Monitor", grid_final)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for c in caps: c.release()
    cv2.destroyAllWindows()