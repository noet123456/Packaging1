import torch
import os
from ultralytics import YOLO

# Load model pre-trained
# The error indicates that the dataset is not in OBB format.
# If your dataset has standard (axis-aligned) bounding boxes, use a regular YOLOv8 detection model.
# If you intend to use OBB, you would need to reformat your dataset to include orientation angles.
model = YOLO('yolov8n-obb.pt') # Changed from yolov8n-obb.pt to yolov8n.pt for standard object detection

# Bắt đầu train
results = model.train(
    data='D:\YOLOv11n\dataset7\data.yaml',
    epochs=100,
    imgsz=640,
    batch=32,
    project='Packaging', # Lưu kết quả thẳng về Drive
    name='ComSt2V' # Tên folder kết quả
)