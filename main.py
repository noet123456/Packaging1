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

# Danh mục linh kiện
class_names_lk = [
    'o trong', 'Cap_be_xam', 'Day_Jumper', 'Module_tui',
    'Cap_USB_trang', 'Cap_den', 'main_board', 'Den_to', 'Den_nho'
]

# Transform cho CNN
cnn_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- CÁC HÀM XỬ LÝ ---

def center_crop(frame, crop_width=640, crop_height=480):
    """Cắt khung hình từ chính giữa để không làm biến dạng hình ảnh"""
    h, w = frame.shape[:2]
    start_x = max(0, w // 2 - crop_width // 2)
    start_y = max(0, h // 2 - crop_height // 2)

    cropped_img = frame[start_y:start_y + crop_height, start_x:start_x + crop_width]

    # Nếu ảnh gốc nhỏ hơn kích thước yêu cầu, resize nhẹ lại cho khớp
    if cropped_img.shape[1] != crop_width or cropped_img.shape[0] != crop_height:
        cropped_img = cv2.resize(cropped_img, (crop_width, crop_height))
    return cropped_img


def extract_detections(results):
    """Trích xuất thông tin từ kết quả YOLO OBB"""
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
    """Phân loại chi tiết linh kiện bằng CNN"""
    x, y, w, h = cv2.boundingRect(poly)
    img_h, img_w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x), max(0, y), min(img_w, x + w), min(img_h, y + h)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return "Trong", 0.0

    img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    input_tensor = cnn_transforms(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    if confidence.item() < 0.5:
        return "Trong", confidence.item()
    return class_names_lk[predicted_idx.item()], confidence.item()


def process_and_draw(frame, model_slot, cnn_model):
    """Hàm tổng hợp: Chạy YOLO -> Chạy CNN -> Vẽ kết quả"""
    results_slot = model_slot.predict(frame, conf=0.5, verbose=False)
    detections = extract_detections(results_slot)

    for slot in detections:
        label_name, conf = predict_cnn(cnn_model, frame, slot['poly'])
        # Màu đỏ cho 'Trong', xanh lá cho linh kiện
        color = (0, 0, 255) if label_name == "Trong" else (0, 255, 0)

        cv2.polylines(frame, [slot['poly']], True, color, 2)
        cx, cy = int(slot['xywhr'][0]), int(slot['xywhr'][1])
        cv2.putText(frame, f"{label_name}", (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return frame


# --- LUỒNG CHÍNH ---

if __name__ == "__main__":
    # 1. Load Models
    m_slot_t1 = YOLO(r"D:\YOLOv11n\test\weights\t1\stage1.pt")
    m_slot_t2 = YOLO(r"D:\YOLOv11n\test\weights\t2\stage2.pt")

    m_cnn = models.mobilenet_v3_small()
    m_cnn.classifier[3] = nn.Linear(m_cnn.classifier[3].in_features, len(class_names_lk))
    m_cnn.load_state_dict(torch.load(r"D:YOLOv11n\test\weights\linhkien.pth", map_location=device))
    m_cnn = m_cnn.to(device).eval()

    # 2. Cấu hình Camera RTSP
    rtsp_urls = [
        "rtsp://admin:pass@192.168.1.10/stream1",
        "rtsp://admin:pass@192.168.1.11/stream1",
        "rtsp://admin:pass@192.168.1.12/stream1",
        "rtsp://admin:pass@192.168.1.13/stream1"
    ]
    caps = [cv2.VideoCapture(url) for url in rtsp_urls]

    # Kích thước mỗi ô trong lưới
    W_CROP, H_CROP = 640, 480

    print("Hệ thống đang chạy. Nhấn 'q' để dừng...")

    while True:
        processed_frames = []

        for i, cap in enumerate(caps):
            ret, frame = cap.read()

            if not ret:
                # Tạo khung hình lỗi nếu không có tín hiệu
                error_frame = np.zeros((H_CROP, W_CROP, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"CAM {i + 1} ERROR", (W_CROP // 4, H_CROP // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                processed_frames.append(error_frame)
                continue

            # Thực hiện Center Crop
            frame_cropped = center_crop(frame, W_CROP, H_CROP)

            # Chọn model phù hợp cho từng tầng (Ví dụ 1-2 dùng T1, 3-4 dùng T2)
            current_model = m_slot_t1 if i < 2 else m_slot_t2

            # Xử lý AI và vẽ
            final_frame = process_and_draw(frame_cropped, current_model, m_cnn)
            processed_frames.append(final_frame)

        # 3. Ghép lưới 2x2
        # Hàng 1: Cam 1 | Cam 2
        row1 = np.concatenate((processed_frames[0], processed_frames[1]), axis=1)
        # Hàng 2: Cam 3 | Cam 4
        row2 = np.concatenate((processed_frames[2], processed_frames[3]), axis=1)
        # Tổng hợp:
        # Row 1
        # Row 2
        grid_display = np.concatenate((row1, row2), axis=0)

        # 4. Hiển thị
        cv2.imshow("Industrial Monitoring System 2x2", grid_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()