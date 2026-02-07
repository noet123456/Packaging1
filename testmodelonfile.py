import cv2
from ultralytics import YOLO

# ---- CẤU HÌNH ----
# Đường dẫn file model
MODEL_PATH = r"D:\YOLOv11n\Packaging\weights\t1\weights\best.pt"
# Đường dẫn file video
VIDEO_PATH = r"D:\YOLOv11n\Packaging\record_20260206_190517.mp4"

def main():
    # 1. Load Model
    print(f"⏳ Đang tải model từ {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Lỗi tải model: {e}")
        return

    # 2. Mở file video
    print(f"⏳ Đang mở video {VIDEO_PATH}...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("❌ Không thể mở file video.")
        return

    # Lấy thông số video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Video info: {width}x{height} @ {fps} FPS")

    # Kích thước resize (tùy chỉnh nếu muốn chạy nhanh hơn hoặc đúng input model)
    new_width, new_height = 640, 640 

    print("▶️ Bắt đầu nhận diện... Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⏹️ Kết thúc video.")
            break

        # Resize nếu cần thiết
        frame = cv2.resize(frame, (new_width, new_height))

        # 3. Nhận diện (Inference)
        # conf=0.5: chỉ lấy kết quả có độ tin cậy > 0.5
        results = model.predict(frame, conf=0.6, verbose=False)

        # 4. Vẽ kết quả
        annotated_frame = results[0].plot()

        # 5. Hiển thị
        cv2.imshow("Test Model on Video", annotated_frame)

        # Chờ 1ms, nếu bấm 'q' thì thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("⏹️ Đã dừng bởi người dùng.")
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
