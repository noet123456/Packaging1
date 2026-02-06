import cv2
import datetime
from ultralytics import YOLO  # Đảm bảo đã cài: pip install ultralytics

# ---- CẤU HÌNH ----
# Đường dẫn file .pt trên máy bạn
MODEL_PATH = r"D:\YOLOv11n\Packaging\weights\t1\best.pt"
# Link RTSP camera
url = "rtsp://admin:BWKUYM@192.168.1.144:554/ch1/main"

# 1. Load Model
print("⏳ Đang tải model AI...")
model = YOLO(MODEL_PATH)

# 2. Mở kết nối camera
cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("❌ Không kết nối được camera.")
    exit()

# ---- FPS & KÍCH THƯỚC ----
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
new_width, new_height = 640, 480
# ---- Vòng lặp xử lý ----
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Mất tín hiệu camera.")
        break

    # 3. RESIZE FRAME trước khi đưa vào AI (hoặc sau tùy bạn, nhưng resize trước để đồng nhất)
    frame_resized = cv2.resize(frame, (new_width, new_height))

    # 4. NHẬN DIỆN (Inference)
    # results trả về danh sách kết quả (thường chỉ lấy kết quả đầu tiên [0])
    results = model.predict(frame_resized, conf=0.5, verbose=False)

    # 5. VẼ KẾT QUẢ lên frame (vẽ box, label)
    # kết quả được vẽ trực tiếp lên một bản sao của frame
    annotated_frame = results[0].plot()

    # 7. Hiển thị
    cv2.imshow("AI Detection & Recording", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---- Giải phóng tài nguyên ----
cap.release()
cv2.destroyAllWindows()
