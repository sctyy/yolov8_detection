import sys
sys.path.append('/mnt/c/Users/29656/Desktop/分割算法/ultralytics-main/')
import ultralytics
print(ultralytics.__file__)
from ultralytics import YOLO
from pybaseutils import font_utils
font_file = font_utils.set_pyplot_font(font="simhei")


if __name__ == "__main__":
    # Load a model
    model_yaml = "/mnt/c/Users/29656/Desktop/分割算法/ultralytics-main/zoo/yolov8s.yaml"
    data_yaml = "/mnt/c/Users/29656/Desktop/分割算法/ultralytics-main/datasets/datasets.yaml"
    pre_model = "/mnt/c/Users/29656/Desktop/分割算法/ultralytics-main/zoo/yolov8s.pt"

    model = YOLO(model_yaml, task='detect').load(pre_model)  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=data_yaml, imgsz=[512, 1024], epochs=500, batch=4, project='/mnt/d/疵点数据库/标注疵点/resize', patience=50, workers=4, device='0')

    # rect=True，启用矩形训练，此时数据预处理会根据接收的imgsz的最大值来保持原始尺寸的宽高比进行短边的缩放，同时mosaic、cutmix等数据增强方法不会再应用（默认只支持正方形训练）
    # 实验发现矩形训练效果没有原始的训练效果好（letterbox已经修改为了直接resize的方式）
    #results = model.train(data=data_yaml, imgsz=[512, 1024], epochs=500, rect=True ,batch=4, project='/mnt/d/疵点数据库/标注疵点/resize', patience=50, workers=4, device='0')