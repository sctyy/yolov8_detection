from ultralytics import YOLO
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    model = YOLO('./runs/detect/train7/weights/best.pt')  # 加载自定义模型

    # 验证模型
    metrics = model.val(conf=0.5, iou=0.5, imgsz=[384, 768], data='../../datasets/datasets.yaml', split='test', device='0')  # 无需参数，数据集和设置记忆
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps
