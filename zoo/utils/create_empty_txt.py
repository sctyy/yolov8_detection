"""
@Project ：ultralytics-main 
@File    ：create_empty_txt.py
@Author  ：sc
@Date    ：2026/1/21 14:40 
@description：
    补齐空白txt
"""
import os
from pathlib import Path


def create_empty_txt(pred_dir, image_dir):

    image_files = {Path(f).stem for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))}
    existing_txt = {Path(f).stem for f in os.listdir(pred_dir) if f.endswith('.txt')}
    missing = image_files - existing_txt
    for stem in missing:
        empty_txt_path = os.path.join(pred_dir, f"{stem}.txt")
        Path(empty_txt_path).touch()  # 创建空文件
        print(f"创建空txt: {empty_txt_path}")

    print(f"完成！补齐 {len(missing)} 个空txt")


if __name__ == '__main__':
    create_empty_txt(pred_dir="labels", image_dir="images")