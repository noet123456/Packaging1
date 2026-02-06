import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image

import os

# --- THIẾT LẬP HỆ THỐNG ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. DANH SÁCH TÊN GỐC (LÚC TRAIN)
# Đây là tên các folder class bạn dùng khi train model CNN
# Thứ tự phải đúng tuyệt đối với lúc train
class_names_original = [
    'Component 9',  # ID 0
    'Component 1',  # ID 1
    'Component 2',  # ID 2
    'Component 3',  # ID 3
    'Component 4',  # ID 4
    'Component 5',  # ID 5
    'Component 6',  # ID 6
    'Component 7',  # ID 7
    'Component 8',
    'Component 10'# ID 8
]

# 2. TỪ ĐIỂN ÁNH XẠ (MAPPING) - QUAN TRỌNG
# Cấu trúc: 'Tên gốc lúc train': 'Tên hiển thị mong muốn'
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
    'Component 10': 'jumper_mau'
}

# 3. CẤU HÌNH YÊU CẦU (DÙNG TÊN MỚI)
# Lưu ý: Ở đây phải điền TÊN HIỂN THỊ (Value trong mapping)
CAM_REQUIREMENTS = {
    # Cam 1: Cần 3 linh kiện
    0: ['Cam_bien_cd1', 'Cam_bien_cd2', 'Cam_bien_cd3'],

    # Cam 2: Cần 2 linh kiện
    1: ['mainboard', 'J-Link JTAG'],

    # Cam 3: Cần 3 linh kiện
    2: ['day_cap_den', 'cap_bet', 'jumper_mau'],

    # Cam 4: Cần 2 linh kiện (Ví dụ)
    3: ['day_cap_trang', 'tui_module']
}

# Transform cho CNN
cnn_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- CÁC HÀM XỬ LÝ ---

def center_crop(frame, crop_width=640, crop_height=480):
    h, w = frame.shape[:2]
    start_x = max(0, w // 2 - crop_width // 2)
    start_y = max(0, h // 2 - crop_height // 2)
    cropped_img = frame[start_y:start_y + crop_height, start_x:start_x + crop_width]
    if cropped_img.shape[1] != crop_width or cropped_img.shape[0] != crop_height:
        cropped_img = cv2.resize(cropped_img, (crop_width, crop_height))
    return cropped_img


def extract_detections(results):
    detections = []
    r = results[0]
    if r.obb is not None:
        boxes_xywhr = r.obb.xywhr.cpu().numpy()
        cls_ids = r.obb.cls.cpu().numpy()
        xyxyxyxy = r.obb.xyxyxyxy.cpu().numpy()
        names = r.names
        for i in range(len(cls_ids)):
            detections.append({
                'cls_id': int(cls_ids[i]),
                'xywhr': boxes_xywhr[i],
                'poly': xyxyxyxy[i].astype(np.int32),
                'label': names[int(cls_ids[i])]
            })
    return detections


def predict_cnn(model, frame, poly):
    x, y, w, h = cv2.boundingRect(poly)
    img_h, img_w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x), max(0, y), min(img_w, x + w), min(img_h, y + h)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return NAME_MAPPING['o trong'], 0.0

    img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    input_tensor = cnn_transforms(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    # Lấy tên gốc từ model
    raw_name = class_names_original[predicted_idx.item()]

    # --- BƯỚC CHUYỂN ĐỔI TÊN ---
    # Lấy tên hiển thị từ từ điển mapping.
    # Nếu không tìm thấy key thì trả về nguyên gốc raw_name
    display_name = NAME_MAPPING.get(raw_name, raw_name)

    if confidence.item() < 0.5:
        return NAME_MAPPING['o trong'], confidence.item()

    return display_name, confidence.item()


def process_and_draw(frame, model_slot, cnn_model, cam_idx):
    results_slot = model_slot.predict(frame, conf=0.5, verbose=False)
    detections = extract_detections(results_slot)
    found_items = []

    for slot in detections:
        label_name, conf = predict_cnn(cnn_model, frame, slot['poly'])

        # Chỉ thêm vào danh sách tìm thấy nếu KHÔNG PHẢI là "Trong"
        empty_label = NAME_MAPPING['o trong']
        if label_name != empty_label and label_name != "Unknown":
            found_items.append(label_name)

        # Màu sắc: Đỏ nếu trống, Xanh lá nếu có đồ
        color = (0, 0, 255) if label_name == empty_label else (0, 255, 0)
        cv2.polylines(frame, [slot['poly']], True, color, 2)

        cx, cy = int(slot['xywhr'][0]), int(slot['xywhr'][1])
        cv2.putText(frame, label_name, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # --- LOGIC KIỂM TRA (Dùng tên mới để so sánh) ---
    required_items = CAM_REQUIREMENTS.get(cam_idx, [])
    missing_items = [item for item in required_items if item not in found_items]
    is_full = (len(missing_items) == 0)

    status_text = "TRUE (DU)" if is_full else "FALSE (THIEU)"
    status_color = (0, 255, 0) if is_full else (0, 0, 255)

    # Vẽ bảng thông báo nền đen
    cv2.rectangle(frame, (0, 0), (350, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"CAM {cam_idx + 1}: {status_text}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    if not is_full:
        missing_str = ", ".join(missing_items)
        cv2.putText(frame, f"Miss: {missing_str}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame


# --- LUỒNG CHÍNH ---

if __name__ == "__main__":
    print("Loading models...")
    # Load YOLO (Giữ nguyên)
    m_slot_t1 = YOLO(r"D:\YOLOv11n\test\weights\t1\stage1.pt")
    m_slot_t2 = YOLO(r"D:\YOLOv11n\test\weights\t2\stage2.pt")

    # Load CNN
    m_cnn = models.mobilenet_v3_small()
    # Chú ý: output layer phải bằng số lượng class gốc (9 class)
    m_cnn.classifier[3] = nn.Linear(m_cnn.classifier[3].in_features, len(class_names_original))
    m_cnn.load_state_dict(torch.load(r"D:\YOLOv11n\test\weights\linhkien.pth", map_location=device))
    m_cnn = m_cnn.to(device).eval()

    # Cấu hình Camera RTSP
    rtsp_urls = [
        "rtsp://admin:CPSFLT@192.168.1.160:554/ch1/main",
        "rtsp://admin:DVCLRQ@192.168.1.116:554/ch1/main",
        "rtsp://admin:BWKUYM@192.168.1.144:554/ch1/main",
        "rtsp://admin:KXILGD@192.168.1.152:554/ch1/main"
    ]
    caps = [cv2.VideoCapture(url) for url in rtsp_urls]
    W_CROP, H_CROP = 640, 480

    print("Hệ thống đang chạy với tính năng Đổi Tên (Mapping)...")

    while True:
        processed_frames = []

        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                error_frame = np.zeros((H_CROP, W_CROP, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"CAM {i + 1} ERROR", (50, H_CROP // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                processed_frames.append(error_frame)
                continue

            frame_cropped = center_crop(frame, W_CROP, H_CROP)
            current_model = m_slot_t1 if i < 2 else m_slot_t2
            final_frame = process_and_draw(frame_cropped, current_model, m_cnn, cam_idx=i)
            processed_frames.append(final_frame)

        if len(processed_frames) == 4:
            row1 = np.concatenate((processed_frames[0], processed_frames[1]), axis=1)
            row2 = np.concatenate((processed_frames[2], processed_frames[3]), axis=1)
            grid_display = np.concatenate((row1, row2), axis=0)
            cv2.imshow("Production Monitor - Mapped Names", grid_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps: cap.release()
    cv2.destroyAllWindows()