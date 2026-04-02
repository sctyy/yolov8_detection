from ultralytics import YOLO

# 加载模型
model = YOLO('/mnt/c/Users/29656/Desktop/yolovs_all_v5_20260401.pt')

# 导出模型
model.export(format='onnx', imgsz=[672, 1024], dynamic=False, simplify=True, opset=12)