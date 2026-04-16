"""
@Project ：ultralytics-main
@File    ：test.py
@Author  ：sc
@Date    ：2025/8/19 16:20
@description：
    yolov8测试脚本
"""

import os
import torch
import cv2
from config import yolov8_yanbuji_mapping, label_yanbuji_color, labels_circle_machine_yolov8, label_color
import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMG_SIZE = (384, 768)
CONF_THRES = 0.35
IOU_THRES = 0.45
MODEL_PATH = r'/mnt/c/Users/29656/Desktop/last.pt'  # 模型路径
IMAGE_PATH = r'/mnt/c/Users/29656/Desktop/test/data2'  # 测试图片路径
SAVE_PATH = r'/mnt/c/Users/29656/Desktop/test/output2'   # 结果保存
RESIZE_MODE = 'resize'         # 数据预处理方法，可以选择 'resize' 或 'letterbox'

id2name = {v: k for k, v in yolov8_yanbuji_mapping.items()}  # 类别名称
id2color = {i: tuple(c) for i, c in enumerate(label_color)}
FONT_PATH = "Simhei.ttf"
FONT = ImageFont.truetype(FONT_PATH, 50)


def letterbox(im, new_shape=IMG_SIZE, color=(114, 114, 114)):
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = int(round(w * r)), int(round(h * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def direct_resize(im, new_shape=IMG_SIZE):
    im = cv2.resize(im, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)
    scale_x = im.shape[1] / new_shape[1]
    scale_y = im.shape[0] / new_shape[0]
    return im, (scale_x, scale_y), (0, 0)


def xywh2xyxy(x):
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def nms(boxes, scores, iou_threshold):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())

        rest_boxes = boxes[order[1:]]
        curr_box = boxes[i].unsqueeze(0)
        inter_x1 = torch.max(curr_box[:, 0], rest_boxes[:, 0])
        inter_y1 = torch.max(curr_box[:, 1], rest_boxes[:, 1])
        inter_x2 = torch.min(curr_box[:, 2], rest_boxes[:, 2])
        inter_y2 = torch.min(curr_box[:, 3], rest_boxes[:, 3])

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        curr_area = (curr_box[:, 2] - curr_box[:, 0]) * (curr_box[:, 3] - curr_box[:, 1])
        rest_area = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (rest_boxes[:, 3] - rest_boxes[:, 1])
        iou = inter_area / (curr_area + rest_area - inter_area)

        mask = iou <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, agnostic=False, max_det=300):
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    prediction = prediction.transpose(-1, -2)
    nc = prediction.shape[1] - 4
    xc = prediction[:, 4:4 + nc].amax(1) > conf_thres
    x = prediction[xc]

    if not x.shape[0]:
        return torch.zeros((0, 6), device=prediction.device)

    x[:, :4] = xywh2xyxy(x[:, :4])

    box = x[:, :4]
    cls = x[:, 4:4 + nc]

    conf, j = cls.max(1, keepdim=True)
    x = torch.cat((box, conf, j.float()), 1)

    x = x[x[:, 4].argsort(descending=True)]

    c = x[:, 5:6] * (0 if agnostic else 7680)  # 类别偏移，类别感知nms
    boxes = x[:, :4] + c
    scores = x[:, 4]

    i = nms(boxes, scores, iou_thres)
    i = i[:max_det]
    return x[i]


def scale_coords(img0_shape, coords, scale_pad):
    if RESIZE_MODE == 'resize':
        sx = img0_shape[1] / IMG_SIZE[1]
        sy = img0_shape[0] / IMG_SIZE[0]
        coords[:, [0, 2]] *= sx
        coords[:, [1, 3]] *= sy
    else:  # letterbox
        gain, pad = scale_pad[0], scale_pad[1]
        coords[:, [0, 2]] -= pad[0]; coords[:, [1, 3]] -= pad[1]
        coords[:, :4] /= gain
    coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, img0_shape[1])
    coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, img0_shape[0])
    return coords


def cv2_add_name(img, text, pos, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    bbox = draw.textbbox((0, 0), text, font=FONT)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([pos[0] - 10, pos[1] - h - 10, pos[0] + w + 10, pos[1] + 10], fill=bg_color)
    draw.text((pos[0], pos[1] - h), text, font=FONT, fill=text_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def save_detection_result(image_path: str, pred_tensor=None, save_path: str = None, gt_dir: str = None, conf_thres: float = 0.25, save_txt: bool = False):

    img = cv2.imread(image_path)
    if img is None:
        print(f"读取失败: {image_path}")
        return
    h, w = img.shape[:2]
    has_gt = False
    gt_path = None
    if gt_dir is not None:
        txt_name = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
        gt_path = os.path.join(gt_dir, txt_name)
        has_gt = os.path.exists(gt_path)

    img_pred = img.copy()
    img_gt = img.copy() if has_gt else None
    if save_txt:
        txt_dir = os.path.join(os.path.dirname(save_path), 'labels')
        os.makedirs(txt_dir, exist_ok=True)
        txt_file = os.path.join(txt_dir, os.path.basename(image_path).replace('.jpg', '.txt'))
        lines = []
    if has_gt:
        with open(gt_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)

                color = id2color.get(cls_id, (255, 0, 0))
                name = id2name.get(cls_id, "未知")

                cv2.rectangle(img_gt, (x1, y1), (x2, y2), color, 5)
                img_gt = cv2_add_name(img_gt, f"{name}", (x1, y1 - 20), text_color=(255, 255, 255), bg_color=color)

    if pred_tensor is not None and len(pred_tensor) > 0:
        for *xyxy, conf, cls in pred_tensor.cpu().numpy():
            if conf < conf_thres: continue
            x1, y1, x2, y2 = map(int, xyxy)
            cls_id = int(cls)
            color = id2color.get(cls_id, (0, 255, 0))
            name = id2name.get(cls_id, "未知")

            cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 5)
            img_pred = cv2_add_name(img_pred, f"{name} {conf:.2f}", (x1, y1 - 20), text_color=(255, 255, 255), bg_color=color)

            if save_txt:
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    if save_txt:
        with open(txt_file, 'w', encoding='utf-8') as f:
            if lines:
                f.write('\n'.join(lines) + '\n')

    if has_gt:
        result = np.hstack([img_gt, img_pred])
    else:
        result = img_pred

    save_path = save_path or os.path.join("results", os.path.basename(image_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, result)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = ckpt.get('ema') or ckpt.get('model')
    model.float().eval()
    for root, dirs, files in os.walk(IMAGE_PATH):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                print(file)
                img0 = cv2.imread(os.path.join(root, file))
                if img0 is None:
                    raise FileNotFoundError(IMAGE_PATH)

                if RESIZE_MODE == 'resize':
                    img, scale_xy, pad = direct_resize(img0)
                else:
                    img, gain, pad = letterbox(img0)
                    scale_xy = (gain, pad)
                img_rgb = img[:, :, ::-1].transpose(2, 0, 1)
                tensor = torch.from_numpy(img_rgb[None].copy()).to(device, dtype=torch.float32) / 255.0

                with torch.no_grad():
                    out = model(tensor)
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    pred = non_max_suppression(out[0], conf_thres=CONF_THRES, iou_thres=IOU_THRES, max_det=300)

                if pred is not None and len(pred):
                    pred[:, :4] = scale_coords(img0.shape[:2], pred[:, :4], scale_xy)

                    for *xyxy, conf, cls in pred.cpu().numpy():
                        x1, y1, x2, y2 = map(int, xyxy)
                        name = id2name.get(cls, "未知")
                        label = f'{name} {conf:.2f}'
                        print(f"图片:{file},检测到：{label},坐标：{x1, y1, x2, y2}")

                save_detection_result(
                    image_path=os.path.join(root, file),
                    pred_tensor=pred,
                    save_path=os.path.join(SAVE_PATH, file),
                    gt_dir=IMAGE_PATH.replace('images', 'labels'),  # 标注的txt默认在images同级目录，用于绘制对比结果图片
                    conf_thres=CONF_THRES,
                    save_txt=True
                )


if __name__ == '__main__':
    main()