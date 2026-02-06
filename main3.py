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

# Danh mục toàn bộ linh kiện (Đảm bảo tên ở đây khớp với tên folder lúc train)
class_names_lk = [
    'o trong', 'Cap_be_xam', 'Day_Jumper', 'Module_tui',
    'Cap_USB_trang', 'Cap_den', 'main_board', 'Den_to', 'Den_nho'
]

# --- CẤU HÌNH YÊU CẦU CHO TỪNG CAMERA (BẠN CẦN SỬA CHỖ NÀY) ---
# Quy định: Cam nào bắt buộc phải có những linh kiện nào?
CAM_REQUIREMENTS = {
    # Cam 1: 3 linh kiện đầu tầng 1
    0: ['Cap_be_xam', 'Day_Jumper', 'Module_tui'],

    # Cam 2: 2 linh kiện cuối tầng 1
    1: ['Cap_USB_trang', 'Cap_den'],

    # Cam 3: 3 linh kiện đầu tầng 2
    2: ['main_board', 'Den_to', 'Den_nho'],

    # Cam 4: 2 linh kiện cuối tầng 2
    3: ['Cap_be_xam', 'Day_Jumper']  # (Ví dụ lặp lại, bạn hãy sửa lại cho đúng thực tế)
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
    if crop.size == 0: return "o trong", 0.0

    img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    input_tensor = cnn_transforms(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    if confidence.item() < 0.5:
        return "o trong", confidence.item()
    return class_names_lk[predicted_idx.item()], confidence.item()


def process_and_draw(frame, model_slot, cnn_model, cam_idx):
    """
    Xử lý nhận diện và kiểm tra Logic Đủ/Thiếu
    """
    results_slot = model_slot.predict(frame, conf=0.5, verbose=False)
    detections = extract_detections(results_slot)

    # Danh sách các linh kiện ĐÃ TÌM THẤY trong khung hình này
    found_items = []

    # 1. Vẽ và nhận diện từng vật thể
    for slot in detections:
        label_name, conf = predict_cnn(cnn_model, frame, slot['poly'])

        # Lưu lại tên linh kiện tìm thấy (trừ ô trống và lỗi)
        if label_name != "o trong" and label_name != "Unknown":
            found_items.append(label_name)

        # Màu sắc ô: Đỏ nếu trống, Xanh lá nếu có đồ
        color = (0, 0, 255) if label_name == "o trong" else (0, 255, 0)
        cv2.polylines(frame, [slot['poly']], True, color, 2)

        # Vẽ tên linh kiện nhỏ cạnh ô
        cx, cy = int(slot['xywhr'][0]), int(slot['xywhr'][1])
        cv2.putText(frame, label_name, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # 2. LOGIC KIỂM TRA ĐỦ/THIẾU (Check Completeness)
    required_items = CAM_REQUIREMENTS.get(cam_idx, [])

    # Tìm những món CẦN nhưng CHƯA THẤY
    # (Dùng logic tập hợp để so sánh)
    missing_items = [item for item in required_items if item not in found_items]

    is_full = (len(missing_items) == 0)

    # 3. Hiển thị kết quả TRUE/FALSE lên góc màn hình
    status_text = "TRUE (DU)" if is_full else "FALSE (THIEU)"
    status_color = (0, 255, 0) if is_full else (0, 0, 255)  # Xanh lá vs Đỏ

    # Vẽ nền đen cho chữ dễ đọc
    cv2.rectangle(frame, (0, 0), (300, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"CAM {cam_idx + 1}: {status_text}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    if not is_full:
        # Nếu thiếu, liệt kê món thiếu
        missing_str = ", ".join(missing_items)
        cv2.putText(frame, f"Miss: {missing_str}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame


# --- LUỒNG CHÍNH ---

if __name__ == "__main__":
    # 1. Load Models (Giữ nguyên đường dẫn của bạn)
    print("Loading models...")
    m_slot_t1 = YOLO(r"D:\YOLOv11n\test\weights\t1\stage1.pt")
    m_slot_t2 = YOLO(r"D:\YOLOv11n\test\weights\t2\stage2.pt")

    m_cnn = models.mobilenet_v3_small()
    m_cnn.classifier[3] = nn.Linear(m_cnn.classifier[3].in_features, len(class_names_lk))
    m_cnn.load_state_dict(torch.load(r"D:\YOLOv11n\test\weights\linhkien.pth", map_location=device))
    m_cnn = m_cnn.to(device).eval()

    # 2. Cấu hình Camera RTSP
    rtsp_urls = [
        "rtsp://admin:pass@192.168.1.10/stream1",  # Cam 1
        "rtsp://admin:pass@192.168.1.11/stream1",  # Cam 2
        "rtsp://admin:pass@192.168.1.12/stream1",  # Cam 3
        "rtsp://admin:pass@192.168.1.13/stream1"  # Cam 4
    ]
    caps = [cv2.VideoCapture(url) for url in rtsp_urls]
    W_CROP, H_CROP = 640, 480

    print("Hệ thống kiểm tra đóng gói đang chạy...")
    print("Logic: TRUE = Đủ linh kiện, FALSE = Thiếu linh kiện")

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

            # Chọn model YOLO theo tầng
            current_model = m_slot_t1 if i < 2 else m_slot_t2

            # GỌI HÀM XỬ LÝ MỚI (Truyền thêm tham số i)
            final_frame = process_and_draw(frame_cropped, current_model, m_cnn, cam_idx=i)
            processed_frames.append(final_frame)

        # 3. Hiển thị lưới 2x2
        if len(processed_frames) == 4:
            row1 = np.concatenate((processed_frames[0], processed_frames[1]), axis=1)
            row2 = np.concatenate((processed_frames[2], processed_frames[3]), axis=1)
            grid_display = np.concatenate((row1, row2), axis=0)
            cv2.imshow("Quality Control System (True/False Check)", grid_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps: cap.release()
    cv2.destroyAllWindows()