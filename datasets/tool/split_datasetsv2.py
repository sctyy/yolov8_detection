"""
   数据集分割
"""
import shutil
import random
import os

# 检查文件夹是否存在
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 将路径写入txt文件
def write_paths_to_txt(file_path, paths):
    with open(file_path, 'w') as f:
        for path in paths:
            # f.write(os.path.relpath(path, start=os.path.dirname(file_path)) + '\n')
            f.write(path + '\n')

def split(image_dir, txt_dir, save_dir):
    # 创建文件夹
    mkdir(save_dir)
    images_dir = os.path.join(save_dir, 'images')
    labels_dir = os.path.join(save_dir, 'labels')

    img_train_path = os.path.join(images_dir, 'train')
    img_test_path = os.path.join(images_dir, 'test')
    img_val_path = os.path.join(images_dir, 'val')

    label_train_path = os.path.join(labels_dir, 'train')
    label_test_path = os.path.join(labels_dir, 'test')
    label_val_path = os.path.join(labels_dir, 'val')

    mkdir(images_dir)
    mkdir(labels_dir)
    mkdir(img_train_path)
    mkdir(img_test_path)
    mkdir(img_val_path)
    mkdir(label_train_path)
    mkdir(label_test_path)
    mkdir(label_val_path)

    # 数据集划分比例，训练集75%，验证集15%，测试集10%，按需修改
    train_percent = 0.75
    val_percent = 0.15
    test_percent = 0.10

    # 获取所有子文件夹
    sub_dirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]

    train_paths = []
    val_paths = []
    test_paths = []

    for sub_dir in sub_dirs:
        # 为每个子文件夹创建对应的路径
        src_sub_img_dir = os.path.join(image_dir, sub_dir)
        src_sub_txt_dir = os.path.join(txt_dir, sub_dir)

        dst_sub_img_train_path = os.path.join(img_train_path, sub_dir)
        dst_sub_img_test_path = os.path.join(img_test_path, sub_dir)
        dst_sub_img_val_path = os.path.join(img_val_path, sub_dir)

        dst_sub_label_train_path = os.path.join(label_train_path, sub_dir)
        dst_sub_label_test_path = os.path.join(label_test_path, sub_dir)
        dst_sub_label_val_path = os.path.join(label_val_path, sub_dir)

        mkdir(dst_sub_img_train_path)
        mkdir(dst_sub_img_test_path)
        mkdir(dst_sub_img_val_path)
        mkdir(dst_sub_label_train_path)
        mkdir(dst_sub_label_test_path)
        mkdir(dst_sub_label_val_path)

        # 获取当前子文件夹下的所有图片
        total_images = os.listdir(src_sub_img_dir)
        num_images = len(total_images)
        list_all_images = range(num_images)

        num_train = int(num_images * train_percent)
        num_val = int(num_images * val_percent)
        num_test = num_images - num_train - num_val

        train_idx = random.sample(list_all_images, num_train)
        val_test_idx = [i for i in list_all_images if not i in train_idx]
        val_idx = random.sample(val_test_idx, num_val)

        for i in list_all_images:
            name = total_images[i]

            srcImage = os.path.join(src_sub_img_dir, name)
            srcLabel = os.path.join(src_sub_txt_dir, os.path.splitext(name)[0] + '.txt')

            if i in train_idx:
                dst_train_Image = os.path.join(dst_sub_img_train_path, name)
                dst_train_Label = os.path.join(dst_sub_label_train_path, os.path.splitext(name)[0] + '.txt')
                shutil.copyfile(srcImage, dst_train_Image)
                shutil.copyfile(srcLabel, dst_train_Label)
                train_paths.append(dst_train_Image)
            elif i in val_idx:
                dst_val_Image = os.path.join(dst_sub_img_val_path, name)
                dst_val_Label = os.path.join(dst_sub_label_val_path, os.path.splitext(name)[0] + '.txt')
                shutil.copyfile(srcImage, dst_val_Image)
                shutil.copyfile(srcLabel, dst_val_Label)
                val_paths.append(dst_val_Image)
            else:
                dst_test_Image = os.path.join(dst_sub_img_test_path, name)
                dst_test_Label = os.path.join(dst_sub_label_test_path, os.path.splitext(name)[0] + '.txt')
                shutil.copyfile(srcImage, dst_test_Image)
                shutil.copyfile(srcLabel, dst_test_Label)
                test_paths.append(dst_test_Image)

    write_paths_to_txt(os.path.join(save_dir, 'train.txt'), train_paths)
    write_paths_to_txt(os.path.join(save_dir, 'val.txt'), val_paths)
    write_paths_to_txt(os.path.join(save_dir, 'test.txt'), test_paths)

if __name__ == '__main__':
    image_dir = '/mnt/c/Users/29656/Desktop/yolo/images'
    txt_dir = '/mnt/c/Users/29656/Desktop/yolo/txt'
    save_dir = '/mnt/c/Users/29656/Desktop/yolo/data'

    split(image_dir, txt_dir, save_dir)