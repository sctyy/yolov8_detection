# coding=UTF-8
"""
   json转换成语义分割的yolov8的txt(归一化之后的)
"""

import os
import json
import cv2
from config import labels_circle_machine_mapping as labels
import numpy as np
# 文件夹路径和输出路径
# input_folder = '/data/data_team/share/算法组训练数据库/all/data/罗纹/jsons/'#json文件地址
# output_folder = '/home/tny/workspace/tnysegnet/datasets/罗纹/'#转换后保存地址

input_dir = '/mnt/c/Users/29656/Desktop/data'
output_dir = '/mnt/c/Users/29656/Desktop/data'

buzhong = ['汗布', '罗纹', '特别布', '粗针罗纹', '粗针华夫格', '粗针汗布', '粗针竹节', '网眼布', '单面人字纹', '华夫格', '空气层', '楼梯布', '罗马布', '罗纹小提花', '罗纹小提花华夫格', '棉毛', '三线卫衣', '色织汗布', '色织空气层',
           '色织-罗纹', '色织圈圈纱', '色织双面仿仿罗纹', '色织条', '色织小提花', '色织小循环', '小毛圈', '小提花', '斜纹布', '直条布', '皱皱布']
# buzhong = ['汗布', '罗纹']
for i in buzhong:
    input_folder = os.path.join(input_dir, i) + '/jsons/'
    output_folder = os.path.join(output_dir, i) + '/labels/'
    print(output_folder)
    # 遍历文件夹中所有的JSON文件
    for root, dirs, files in os.walk(input_folder, topdown=True):
        for filename in files:
            if filename.endswith('.json'):
                json_path = os.path.join(root, filename)
                print(json_path)
                txt_path = root.replace(input_folder, output_folder)
                output_txt_path = os.path.join(txt_path, filename.replace('.json', '.txt'))
                if os.path.exists(output_txt_path):
                    print(f"跳过已存在的文件: {output_txt_path}")
                    continue
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                image_path = json_path.replace('jsons', 'images').replace('.json', '.jpg')
                image = cv2.imread(image_path)
                image_width = image.shape[1]
                image_height = image.shape[0]

                if data['shapes'] is None or len(data['shapes']) == 0:
                    # 生成空的txt文件
                    # output_txt_path = json_path.replace('jsons', 'txt').replace('.json', '.txt')
                    txt_path = root.replace(input_folder, output_folder)
                    if not os.path.exists(txt_path):
                        os.mkdir(txt_path)
                    output_txt_path = os.path.join(txt_path, filename.replace('.json', '.txt'))
                    with open(output_txt_path, 'w') as out_file:
                        pass  # 写入空内容

                else:
                    for shape in data['shapes']:
                        points = shape['points']
                        if 'label' in shape:
                            label = shape['label']
                            if points is None or label not in labels:
                                # 生成空的txt文件
                                txt_path = root.replace(input_folder, output_folder)
                                if not os.path.exists(txt_path):
                                    os.mkdir(txt_path)
                                output_txt_path = os.path.join(txt_path, filename.replace('.json', '.txt'))
                                with open(output_txt_path, 'w') as out_file:
                                    pass  # 写入空内容
                            else:
                                # 计算最小和最大x, y
                                class_id = labels[label] - 1
                                #normalized_points = [(x / image_width, y / image_height) for x, y in points]
                                normalized_points = [(x / 1, y / 1) for x, y in points]
                                txt_path = root.replace(input_folder, output_folder)
                                if not os.path.exists(txt_path):
                                   os.mkdir(txt_path)
                                output_txt_path = os.path.join(txt_path, filename.replace('.json', '.txt'))
                                with open(output_txt_path, 'a') as out_file:
                                    out_file.write(f"{class_id}")
                                    for point in normalized_points:
                                        out_file.write(f" {point[0]:.6f} {point[1]:.6f} ")
                                    out_file.write(f"\n")
                        else:
                            txt_path = root.replace(input_folder, output_folder)
                            if not os.path.exists(txt_path):
                                os.mkdir(txt_path)
                            output_txt_path = os.path.join(txt_path, filename.replace('.json', '.txt'))
                            with open(output_txt_path, 'w') as out_file:
                                pass  # 写入空内容

print("转换完成！")
