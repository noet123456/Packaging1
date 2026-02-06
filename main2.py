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

# --- QUAN TRỌNG: TÁCH RIÊNG DANH SÁCH LINH KIỆN 2 TẦNG ---
# Giả sử Tầng 1 và Tầng 2 có linh kiện khác nhau
# (Nếu giống nhau thì để 2 list y hệt nhau cũng được)
class_names_T1 = [
    'o trong', 'Cap_be_xam', 'Day_Jumper', 'Module_tui', 'Cap_USB_trang'
]

class_names_T2 = [
    'o trong', 'Cap_den', 'main_board', 'Den_to', 'Den_nho', 'oc_vit_nho'
]

# Transform cho CNN (Giữ nguyên)
cnn_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- HÀM LOAD MODEL CNN (Viết hàm để đỡ phải copy paste code) ---
def load_cnn_model(pth_path, class_list):
    print(f"Loading CNN from {pth_path}...")
    model = models.mobilenet_v3_small()
    # Sửa lớp cuối cùng cho khớp số lượng class của từng tầng
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_list))

    # Load trọng số
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model = model.to(device).eval()

    # Mẹo: Gắn luôn danh sách tên vào model để tiện dùng sau này
    model.class_names = class_list
    return model


# --- CÁC HÀM XỬ LÝ (SỬA ĐỔI) ---

def center_crop(frame, crop_width=640, crop_height=480):
    # (Giữ nguyên như code cũ của bạn)
    h, w = frame.shape[:2]
    start_x = max(0, w // 2 - crop_width // 2)
    start_y = max(0, h // 2 - crop_height // 2)
    cropped_img = frame[start_y:start_y + crop_height, start_x:start_x + crop_width]
    if cropped_img.shape[1] != crop_width or cropped_img.shape[0] != crop_height:
        cropped_img = cv2.resize(cropped_img, (crop_width, crop_height))
    return cropped_img


def extract_detections(results):
    # (Giữ nguyên như code cũ của bạn)
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
    """Phân loại chi tiết - Đã sửa để dùng class_names riêng của từng model"""
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

    if confidence.item() < 0.5:  # Ngưỡng tin cậy
        return "Unknown", confidence.item()

    # QUAN TRỌNG: Lấy tên từ danh sách riêng của model đó
    predicted_label = model.class_names[predicted_idx.item()]
    return predicted_label, confidence.item()


def process_and_draw(frame, model_slot, cnn_model):
    # (Giữ nguyên logic, chỉ thay đổi luồng dữ liệu bên trong)
    results_slot = model_slot.predict(frame, conf=0.5, verbose=False)
    detections = extract_detections(results_slot)

    for slot in detections:
        # Truyền đúng cnn_model của tầng đó vào
        label_name, conf = predict_cnn(cnn_model, frame, slot['poly'])

        color = (0, 0, 255) if label_name in ["Trong", "Unknown"] else (0, 255, 0)

        cv2.polylines(frame, [slot['poly']], True, color, 2)
        cx, cy = int(slot['xywhr'][0]), int(slot['xywhr'][1])

        # Hiển thị tên linh kiện + độ tin cậy
        cv2.putText(frame, f"{label_name}", (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return frame


# --- LUỒNG CHÍNH ---

if __name__ == "__main__":
    # 1. Load YOLO Models (2 file .pt)
    print("Loading YOLO models...")
    yolo_t1 = YOLO(r"D:\YOLOv11n\test\weightT1.pt")
    yolo_t2 = YOLO(r"D:\YOLOv11n\test\weightT2.pt")

    # 2. Load CNN Models (2 file .pth riêng biệt)
    # Lưu ý: Thay đường dẫn file .pth tương ứng của bạn vào đây
    print("Loading CNN classifiers...")
    cnn_t1 = load_cnn_model(r"D:\YOLOv11n\test\weights\t1\stage1.pth", class_names_T1)
    cnn_t2 = load_cnn_model(r"D:\YOLOv11n\test\weights\t2\stage2.pth", class_names_T2)

    # 3. Cấu hình Camera
    rtsp_urls = [
        "rtsp://admin:pass@192.168.1.10/stream1",  # Cam 1 (Tầng 1)
        "rtsp://admin:pass@192.168.1.11/stream1",  # Cam 2 (Tầng 1)
        "rtsp://admin:pass@192.168.1.12/stream1",  # Cam 3 (Tầng 2)
        "rtsp://admin:pass@192.168.1.13/stream1"  # Cam 4 (Tầng 2)
    ]
    caps = [cv2.VideoCapture(url) for url in rtsp_urls]
    W_CROP, H_CROP = 640, 480

    print("Hệ thống Dual-Model đang chạy...")

    while True:
        processed_frames = []

        for i, cap in enumerate(caps):
            ret, frame = cap.read()

            if not ret:
                error_frame = np.zeros((H_CROP, W_CROP, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"CAM {i + 1} DISCONNECTED", (50, H_CROP // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                processed_frames.append(error_frame)
                continue

            frame_cropped = center_crop(frame, W_CROP, H_CROP)

            # --- LOGIC CHỌN CẶP MODEL (Pairing Logic) ---
            if i < 2:
                # Camera 1 & 2 -> Dùng bộ Tầng 1
                curr_yolo = yolo_t1
                curr_cnn = cnn_t1
            else:
                # Camera 3 & 4 -> Dùng bộ Tầng 2
                curr_yolo = yolo_t2
                curr_cnn = cnn_t2

            # Xử lý
            final_frame = process_and_draw(frame_cropped, curr_yolo, curr_cnn)

            # Thêm nhãn để biết đang chạy model nào (Debug)
            cv2.putText(final_frame, f"Mode: {'T1' if i < 2 else 'T2'}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            processed_frames.append(final_frame)

        # 4. Hiển thị (Giữ nguyên)
        if len(processed_frames) == 4:
            row1 = np.concatenate((processed_frames[0], processed_frames[1]), axis=1)
            row2 = np.concatenate((processed_frames[2], processed_frames[3]), axis=1)
            grid = np.concatenate((row1, row2), axis=0)
            cv2.imshow("Multi-Model System", grid)
        else:
            # Fallback nếu số lượng cam thay đổi
            cv2.imshow("Main", processed_frames[0])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps: cap.release()
    cv2.destroyAllWindows()