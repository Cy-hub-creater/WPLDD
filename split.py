import os
import random
import shutil


def simple_split_dataset(data_root):
    """
    简单版本的数据集分割
    """
    # 创建目标目录
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)

    # 获取所有类别（只包含真正的类别目录）
    all_items = os.listdir(data_root)
    classes = []

    for item in all_items:
        item_path = os.path.join(data_root, item)
        # 只处理目录，并且排除已知的非类别目录
        if os.path.isdir(item_path) and item not in ['train', 'val', 'test', '__pycache__']:
            # 检查是否是文件扩展名（可能是误判的文件）
            if '.' not in item or len(item.split('.')[-1]) > 5:  # 假设扩展名不超过5个字符
                classes.append(item)

    print(f"找到类别: {classes}")

    for class_name in classes:
        print(f"处理: {class_name}")
        class_path = os.path.join(data_root, class_name)

        # 获取所有图像文件
        images = []
        for file in os.listdir(class_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                images.append(file)

        if not images:
            continue

        random.shuffle(images)

        # 分割
        total = len(images)
        train_count = int(total * 0.7)
        val_count = int(total * 0.15)

        train_imgs = images[:train_count]
        val_imgs = images[train_count:train_count + val_count]
        test_imgs = images[train_count + val_count:]

        # 复制文件
        for split, imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            split_class_dir = os.path.join(data_root, split, class_name)
            if not os.path.exists(split_class_dir):
                os.makedirs(split_class_dir)

            for img in imgs:
                src = os.path.join(class_path, img)
                dst = os.path.join(split_class_dir, img)
                shutil.copy2(src, dst)

        print(f"  {class_name}: 训练{len(train_imgs)}, 验证{len(val_imgs)}, 测试{len(test_imgs)}")


if __name__ == "__main__":
    dataset_path = r"D:\PythonProject6\ResNet\wheat_photos"
    simple_split_dataset(dataset_path)
    print("分割完成!")