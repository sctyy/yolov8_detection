# coding=UTF-8
"""
   json转换成目标检测yolov8的txt(归一化之后的)
"""

import os
import json

import cv2

from config import labels_circle_machine_mapping as labels
import numpy as np
# 文件夹路径和输出路径
# input_folder = '/data/data_team/share/算法组训练数据库/all/data/罗纹/jsons/'#json文件地址
# output_folder = '/home/tny/workspace/tnysegnet/datasets/罗纹/'#转换后保存地址

input_dir = '/mnt/d/验布机数据集/data6/jsons'
output_dir = '/mnt/d/验布机数据集/data6/labels'
buzhong = ['汗布']
# buzhong = ['粗针罗纹', '粗针竹节', '单面人字纹', '华夫格', '空气层', '楼梯布', '罗马布', '罗纹小提花', '罗纹小提花华夫格', '棉毛', '三线卫衣', '色织汗布', '色织空气层',
#            '色织-罗纹', '色织圈圈纱', '色织双面仿仿罗纹', '色织条', '色织小提花', '色织小循环', '小毛圈', '小提花', '斜纹布', '直条布', '皱皱布']
for i in buzhong:
    # input_folder = os.path.join(input_dir, i) + '/jsons/'
    # output_folder = os.path.join(output_dir, i) + '/'
    input_folder = input_dir
    output_folder = output_dir
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
                                # output_txt_path = json_path.replace('jsons', 'txt').replace('.json', '.txt')
                                txt_path = root.replace(input_folder, output_folder)
                                if not os.path.exists(txt_path):
                                    os.mkdir(txt_path)
                                output_txt_path = os.path.join(txt_path, filename.replace('.json', '.txt'))
                                with open(output_txt_path, 'w') as out_file:
                                    pass  # 写入空内容
                            else:
                                # 计算最小和最大x, y
                                x_coords = [p[0] for p in points]
                                y_coords = [p[1] for p in points]
                                xmin, xmax = min(x_coords), max(x_coords)
                                ymin, ymax = min(y_coords), max(y_coords)


                                x_center = (xmin + xmax) / 2 / image_width
                                y_center = (ymin + ymax) / 2 / image_height
                                width = (xmax - xmin) / image_width
                                height = (ymax - ymin) / image_height
                                x_center = np.clip(x_center, 0, 1)
                                y_center = np.clip(y_center, 0, 1)
                                width = np.clip(width, 0, 1)
                                height = np.clip(height, 0, 1)
                                # if label not in name:
                                #     name.append(label)
                                # class_id = name.index(label)
                                class_id = labels[label] - 1
                                txt_path = root.replace(input_folder, output_folder)
                                if not os.path.exists(txt_path):
                                   os.mkdir(txt_path)
                                output_txt_path = os.path.join(txt_path, filename.replace('.json', '.txt'))
                                line = f"{class_id} {x_center} {y_center} {width} {height}"
                                existing_lines = set()
                                if os.path.exists(output_txt_path):
                                    with open(output_txt_path, 'r') as in_file:
                                        for existing_line in in_file:
                                            existing_lines.add(existing_line.strip())
                                    # 如果行内容不存在于现有内容中，则写入文件
                                if line not in existing_lines:
                                    with open(output_txt_path, 'a') as out_file:
                                        out_file.write(line + "\n")
                        else:
                            txt_path = root.replace(input_folder, output_folder)
                            if not os.path.exists(txt_path):
                                os.mkdir(txt_path)
                            output_txt_path = os.path.join(txt_path, filename.replace('.json', '.txt'))
                            with open(output_txt_path, 'w') as out_file:
                                pass  # 写入空内容

print("转换完成！")
