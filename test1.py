import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os

# --- THIẾT LẬP HỆ THỐNG ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

CAM_REQUIREMENTS = {
    0: ['Cam_bien_cd1', 'Cam_bien_cd2', 'Cam_bien_cd3'],
    1: ['mainboard', 'J-Link JTAG'],
    2: ['day_cap_den', 'cap_bet', 'jumper_mau'],
    3: ['day_cap_trang', 'tui_module']
}


# --- CÁC HÀM XỬ LÝ ---

def extract_detections(results):
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
        raw_name = model.names[idx]
        return NAME_MAPPING.get(raw_name, raw_name), conf

    return NAME_MAPPING['o trong'], 0.0


# --- LUỒNG CHÍNH ---

if __name__ == "__main__":
    print("Đang khởi tạo hệ thống giám sát...")

    # 1. Load YOLO tìm slot
    m_slot_t1 = YOLO(r"D:\YOLOv11n\test\weights\t1\stage1.pt")
    m_slot_t2 = YOLO(r"D:\YOLOv11n\test\weights\t2\stage2.pt")

    # 2. Load YOLO nhận diện linh kiện
    m_comp = YOLO(r"D:\YOLOv11n\test\weights\t1\stage1.pt")
    checkpoint = torch.load(r"D:\YOLOv11n\test\weights\t1\stage1.pth", map_location=device)

    if 'stage1' in checkpoint:
        try:
            m_comp.model.load_state_dict(checkpoint['stage1'], strict=True)
            print("Nạp trọng số linh kiện thành công.")
        except RuntimeError:
            print("Phát hiện lệch kiến trúc! Đang dùng chế độ tương thích...")
            m_comp.model.load_state_dict(checkpoint['stage1'], strict=False)

        custom_names = checkpoint.get('model_1_names', {})
        if custom_names:
            m_comp.model.names = custom_names

    m_comp.to(device)

    # 3. Cấu hình Camera
    rtsp_urls = [
        "rtsp://admin:CPSFLT@192.168.1.160:554/ch1/main",
        "rtsp://admin:DVCLRQ@192.168.1.116:554/ch1/main",
        "rtsp://admin:BWKUYM@192.168.1.144:554/ch1/main",
        "rtsp://admin:KXILGD@192.168.1.152:554/ch1/main"
    ]
    caps = [cv2.VideoCapture(url) for url in rtsp_urls]

    # THAY ĐỔI: Độ phân giải xử lý (Scale khung hình về mức này để AI chạy nhanh)
    # Nếu muốn nhìn rộng hơn, ta không dùng center_crop mà resize cả ảnh gốc
    W_PROC, H_PROC = 640, 480

    cv2.namedWindow("Production Monitor", cv2.WINDOW_NORMAL)

    while True:
        processed_frames = []
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                err = np.zeros((H_PROC, W_PROC, 3), np.uint8)
                cv2.putText(err, f"CAM {i + 1} LOST", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                processed_frames.append(err)
                continue

            # THAY ĐỔI: Resize thay vì Crop để lấy TOÀN BỘ KHUNG HÌNH rộng
            frame_resized = cv2.resize(frame, (W_PROC, H_PROC))

            m_slot = m_slot_t1 if i < 2 else m_slot_t2
            res_slot = m_slot.predict(frame_resized, conf=0.5, verbose=False)
            slots = extract_detections(res_slot)
            found_items = []

            for s in slots:
                name, conf = predict_component_yolo(m_comp, frame_resized, s['poly'])
                if name != NAME_MAPPING['o trong']:
                    found_items.append(name)

                color = (0, 255, 0) if name != NAME_MAPPING['o trong'] else (0, 0, 255)
                cv2.polylines(frame_resized, [s['poly']], True, color, 2)
                cv2.putText(frame_resized, f"{name}", (s['poly'][0][0], s['poly'][0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            req = CAM_REQUIREMENTS.get(i, [])
            miss = [x for x in req if x not in found_items]
            is_ok = len(miss) == 0

            cv2.rectangle(frame_resized, (0, 0), (320, 65), (0, 0, 0), -1)
            st_text = "OK" if is_ok else "MISSING"
            st_color = (0, 255, 0) if is_ok else (0, 0, 255)
            cv2.putText(frame_resized, f"CAM {i + 1}: {st_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, st_color, 2)
            if not is_ok:
                cv2.putText(frame_resized, f"Need: {', '.join(miss)}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            processed_frames.append(frame_resized)

        if len(processed_frames) == 4:
            top_row = np.hstack((processed_frames[0], processed_frames[1]))
            bottom_row = np.hstack((processed_frames[2], processed_frames[3]))
            full_grid = np.vstack((top_row, bottom_row))

            # PHÓNG ĐẠI HIỂN THỊ (Dành cho màn hình máy tính)
            scale = 1
            grid_final = cv2.resize(full_grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Production Monitor", grid_final)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for c in caps: c.release()
    cv2.destroyAllWindows()