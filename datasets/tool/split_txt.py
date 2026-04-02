"""
YOLOv8 标签按类随机抽取验证/测试集
"""
import os
import random
from pathlib import Path
import argparse
from typing import List

NUM_CLASSES = 17      # 类别数
RANDOM_SEED = 42          


def parse_yolo_label(label_path):
    if not label_path.exists():
        return []
    with open(label_path, encoding='utf-8') as f:
        cls_set = {int(line.split()[0]) for line in f if line.strip()}
    return sorted(cls_set)


def build_dataset(root):
    images_dir = root / 'images'
    labels_dir = root / 'labels'
    dataset = []
    for img in images_dir.rglob('*'):
        if img.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        lbl = labels_dir / img.relative_to(images_dir).with_suffix('.txt')
        print(lbl)
        cls_list = parse_yolo_label(lbl)
        dataset.append((img.resolve(), cls_list))
    return dataset


def split_dataset(dataset, val_num, test_num):
    random.seed(RANDOM_SEED)
    cls_to_imgs = [[] for _ in range(NUM_CLASSES)]
    for img, cls_list in dataset:
        for c in cls_list:
            cls_to_imgs[c].append(img)

    val_set, test_set = set(), set()
    for c in range(NUM_CLASSES):
        imgs = cls_to_imgs[c]
        if len(imgs) < val_num + test_num:
            val_samples = random.sample(imgs, min(val_num, len(imgs)))
            remain = [x for x in imgs if x not in val_samples]
            test_samples = random.sample(remain, min(test_num, len(remain)))
        else:
            val_samples = random.sample(imgs, val_num)
            remain = [x for x in imgs if x not in val_samples]
            test_samples = random.sample(remain, test_num)
        val_set.update(val_samples)
        test_set.update(test_samples)

    train_set = [img for img, _ in dataset if img not in val_set and img not in test_set]
    return train_set, sorted(val_set), sorted(test_set)


def write_txt(imgs, txt_path):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, 'w', encoding='utf-8') as f:
        for img in imgs:
            f.write(str(img) + '\n')


def main():
    parser = argparse.ArgumentParser(description='训练验证测试集txt生成')
    parser.add_argument('--root', default='/data4/yanbuji/data', help='数据集根目录')
    parser.add_argument('--val',  type=int, default=500, help='每类验证集张数')
    parser.add_argument('--test', type=int, default=500, help='每类测试集张数')
    parser.add_argument('--out', default='/data4/yanbuji', help='输出目录')
    args = parser.parse_args()

    root = Path(args.root).resolve()
    dataset = build_dataset(root)
    train, val, test = split_dataset(dataset, args.val, args.test)

    out_path = Path(args.out)
    write_txt(train, out_path / 'train.txt')
    write_txt(val, out_path / 'val.txt')
    write_txt(test, out_path / 'test.txt')

    print(f'训练集: {len(train)} 张')
    print(f'验证集: {len(val)} 张')
    print(f'测试集: {len(test)} 张')
    print('txt生成到了:', out_path.resolve())

if __name__ == '__main__':
    main()

