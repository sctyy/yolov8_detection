from ultralytics import YOLO
model = YOLO('/mnt/c/Users/29656/Desktop/yolov8s_luowen_384_768_resize_20251208.pt')

model.export(format='torchscript',  imgsz=[384, 768], dynamic=False, simplify=True, opset=12)
