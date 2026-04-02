import os
import shutil

# 定义源文件夹和目标文件夹
src_folder = '/mnt/c/Users/29656/Desktop/all'
txt_file = '/mnt/c/Users/29656/Desktop/train.txt'  # 这里替换为你的txt文件名
dst_folder = '/mnt/c/Users/29656/Desktop/train_jsons'

# 确保目标文件夹存在
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

# 读取txt文件中的每行，每行是一个图片路径
with open(txt_file, 'r') as f:
    for line in f:
        line = line.strip()  # 去除两端的空白字符
        if line:  # 确保不处理空行
            # 移除路径中的'images'部分
            img_path_without_images = line.replace('images', '')
            # 获取图片的文件名（包括扩展名）
            img_name = os.path.basename(img_path_without_images)
            # 获取图片的目录结构
            img_dir = os.path.dirname(img_path_without_images)
            # 获取对应的json文件名
            json_name = img_name.replace('.jpg', '.json')
            # 获取源json文件的路径
            src_json_path = os.path.join(src_folder, img_dir, json_name)
            # 构建目标json文件的路径
            dst_json_path = os.path.join(dst_folder, img_dir, json_name)

            # 确保目标目录存在
            dst_json_dir = os.path.dirname(dst_json_path)
            if not os.path.exists(dst_json_dir):
                os.makedirs(dst_json_dir)

            # 复制json文件到目标路径
            shutil.copy(src_json_path, dst_json_path)
            print(f'Copied {src_json_path} to {dst_json_path}')