import json
import os
import cv2
import numpy as np


def convert_labelme_to_yolo_obb(json_path, output_dir, class_mapping):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_w = data['imageWidth']
    img_h = data['imageHeight']
    file_name = os.path.splitext(os.path.basename(json_path))[0]
    output_path = os.path.join(output_dir, f"{file_name}.txt")

    with open(output_path, 'w') as f_out:
        for shape in data['shapes']:
            label = shape['label']
            if label not in class_mapping:
                continue

            class_id = class_mapping[label]
            points = np.array(shape['points'], dtype=np.float32)

            # Tìm hình chữ nhật xoay tối thiểu (Minimum Area Rectangle)
            rect = cv2.minAreaRect(points)
            # Lấy 4 góc của hình chữ nhật
            box = cv2.boxPoints(rect)

            # Chuẩn hóa tọa độ về khoảng [0, 1]
            flatten_box = []
            for p in box:
                norm_x = p[0] / img_w
                norm_y = p[1] / img_h
                flatten_box.extend([norm_x, norm_y])

            # Ghi vào file theo chuẩn: class x1 y1 x2 y2 x3 y3 x4 y4
            line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in flatten_box])
            f_out.write(line + "\n")


# --- Cấu hình ---
input_json_folder = r"E:\Downloads\2(việt)-20260206T100544Z-1-001\2(việt)\json"
output_txt_folder = r"E:\Downloads\2(việt)-20260206T100544Z-1-001\2(việt)\labels"
classes = {"Slot 1": 0, "Slot 2": 1, "Slot 3": 2, "Slot 4": 3, "Slot 5": 4, "Component 1": 5, "Component 2": 6, "Component 3": 7, "Component 4": 8, "Component 5": 9}  # Thay đổi theo labels của bạn

if not os.path.exists(output_txt_folder):
    os.makedirs(output_txt_folder)

for file in os.listdir(input_json_folder):
    if file.endswith(".json"):
        convert_labelme_to_yolo_obb(
            os.path.join(input_json_folder, file),
            output_txt_folder,
            classes
        )
print("Chuyển đổi hoàn tất!")