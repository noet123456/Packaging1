import os
import json
import glob

# ==================================================
# CẤU HÌNH (BẠN CHỈ CẦN SỬA PHẦN NÀY)
# ==================================================

# 1. Đường dẫn đến thư mục chứa file json và ảnh của bạn
# Ví dụ trên Colab: "/content/dataset/train/images"
INPUT_DIR = r"D:\YOLOv11n\test\Tầng 1 (  Sơn )-20260204T023003Z-3-001\Tầng 1 (  Sơn )\image"

# 2. Danh sách Class - PHẢI ĐÚNG THỨ TỰ BẠN MUỐN
# Đây chính là cái sẽ quyết định: box=0, tray=1, gear=2...
# Hãy điền chính xác tên (label) bạn đã gõ trong Labelme
CLASSES = [
    "Slot 1",  # ID 0
    "Slot 2",  # ID 1
    "Slot 3",  # ID 2
    "Slot 4",  # ID 3
    "Slot 5",
    "Component 1",
    "Component 2",
    "Component 3",
    "Component 4",
    "Component 5"# ID 4
]


# ==================================================
# HÀM XỬ LÝ (KHÔNG CẦN SỬA)
# ==================================================

def convert_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
print(f"Tìm thấy {len(json_files)} file JSON. Đang xử lý...")

for json_file in json_files:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_width = data['imageWidth']
    image_height = data['imageHeight']
    txt_content = ""

    for shape in data['shapes']:
        label = shape['label']

        # Nếu tên label không có trong danh sách CLASSES thì bỏ qua hoặc báo lỗi
        if label not in CLASSES:
            print(f"Cảnh báo: Label '{label}' trong file {json_file} không có trong danh sách CLASSES. Đã bỏ qua.")
            continue

        cls_id = CLASSES.index(label)
        points = shape['points']

        # Chuyển đổi từ Polygons (đa giác) hoặc Rect sang Bounding Box (x_min, y_min, x_max, y_max)
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)

        # Tính toán chuẩn hóa YOLO
        bbox = convert_to_yolo((image_width, image_height), (xmin, xmax, ymin, ymax))

        # Ghi vào dòng: class_id x y w h
        txt_content += f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"

    # Lưu file .txt cùng tên với file .json
    txt_filename = json_file.replace(".json", ".txt")
    with open(txt_filename, "w", encoding='utf-8') as f:
        f.write(txt_content)

print("Xong! Đã tạo các file .txt tương ứng.")
print("-" * 30)
print("HÃY COPY ĐOẠN NÀY VÀO FILE data.yaml CỦA BẠN:")
print(f"nc: {len(CLASSES)}")
print("names:")
for idx, name in enumerate(CLASSES):
    print(f"  {idx}: {name}")