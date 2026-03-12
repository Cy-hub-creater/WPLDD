import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import resnet50


def predict_image(img_path, model, device, data_transform, class_indict):
    """
    Predict the class of a single image and return all probabilities.
    """
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    # Apply transformations
    img = data_transform(img)
    # Expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # Prediction
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    return predict_cla, predict[predict_cla].numpy(), predict.numpy()


def predict_folder(folder_path, model, device, data_transform, class_indict, true_labels):
    """
    Predict the class for all images in a folder and calculate accuracy.
    """
    # List all image files in the folder
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    image_files = []
    true_class_labels = []
    true_class_indices = []

    print("收集图像文件和真实标签...")
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)

        # 找到对应的类别索引
        true_class_idx = None
        for idx, class_name in class_indict.items():
            if class_name == subfolder:
                true_class_idx = int(idx)
                break

        if true_class_idx is None:
            print(f"警告: 无法找到类别 '{subfolder}' 的索引，跳过该文件夹")
            continue

        # 获取所有图像文件
        for f in os.listdir(subfolder_path):
            if os.path.isfile(os.path.join(subfolder_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(subfolder_path, f))
                true_class_labels.append(subfolder)
                true_class_indices.append(true_class_idx)

    results = []
    correct_predictions = 0
    total_predictions = len(image_files)

    print(f"开始预测 {total_predictions} 张图像...")
    for idx, img_file in enumerate(image_files):
        img_path = img_file
        if (idx + 1) % 50 == 0:  # 每50张图像打印一次进度
            print(f"Processing: {idx + 1}/{total_predictions}")

        # Predict - 现在返回所有类别的概率
        predict_cla, prob, all_probs = predict_image(img_path, model, device, data_transform, class_indict)

        # Get true label
        true_label = true_class_labels[idx]
        true_cla_idx = true_class_indices[idx]

        # Check if prediction is correct
        is_correct = (predict_cla == true_cla_idx)
        if is_correct:
            correct_predictions += 1

        # Store results in dictionary for easier processing
        results.append({
            'image_path': img_path,
            'predicted_class': class_indict[str(predict_cla)],
            'predicted_probability': prob,
            'true_class': true_label,
            'true_class_idx': true_cla_idx,
            'is_correct': is_correct,
            'all_probabilities': all_probs  # 存储所有类别的概率
        })

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return results, accuracy


def save_four_column_format(results, class_indict, output_file="four_columns_results.txt"):
    """
    保存为四列格式：真实类别 | 预测类别 | 预测概率 | 是否正确
    """
    # 按真实类别分组
    class_groups = {}
    for class_name in class_indict.values():
        class_groups[class_name] = []

    for result in results:
        true_class = result['true_class']
        class_groups[true_class].append(result)

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入标题
        f.write("=" * 60 + "\n")
        f.write("预测结果汇总 (四列格式)\n")
        f.write("=" * 60 + "\n")
        f.write("真实类别\t预测类别\t预测概率\t是否正确\n")
        f.write("-" * 60 + "\n")

        # 按类别顺序写入数据
        for class_name in class_indict.values():
            if class_name in class_groups and class_groups[class_name]:
                f.write(f"\n--- {class_name} ---\n")
                for result in class_groups[class_name]:
                    true_class = result['true_class']
                    predicted_class = result['predicted_class']
                    probability = f"{result['predicted_probability']:.6f}"
                    is_correct = "是" if result['is_correct'] else "否"

                    f.write(f"{true_class}\t{predicted_class}\t{probability}\t{is_correct}\n")

        # 写入统计信息
        f.write("\n" + "=" * 60 + "\n")
        f.write("统计信息\n")
        f.write("=" * 60 + "\n")

        total_images = len(results)
        correct_predictions = sum(1 for r in results if r['is_correct'])
        accuracy = correct_predictions / total_images if total_images > 0 else 0

        f.write(f"总图像数: {total_images}\n")
        f.write(f"正确预测: {correct_predictions}\n")
        f.write(f"错误预测: {total_images - correct_predictions}\n")
        f.write(f"总体准确率: {accuracy:.2%}\n")

        # 各类别统计
        f.write("\n各类别准确率:\n")
        for class_name in class_indict.values():
            class_results = [r for r in results if r['true_class'] == class_name]
            if class_results:
                class_correct = sum(1 for r in class_results if r['is_correct'])
                class_accuracy = class_correct / len(class_results)
                f.write(f"{class_name}: {class_accuracy:.2%} ({class_correct}/{len(class_results)})\n")

    print(f"四列格式结果已保存到: {output_file}")
    return output_file


def save_probability_table(results, class_indict, output_file="probability_table.txt"):
    """
    保存概率表格，每个类别为一列
    """
    class_names = list(class_indict.values())

    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入表头
        header = ["图像名称", "真实类别", "预测类别", "预测概率", "是否正确"]
        header.extend(class_names)
        f.write("\t".join(header) + "\n")

        # 写入数据
        for result in results:
            row = [
                os.path.basename(result['image_path']),
                result['true_class'],
                result['predicted_class'],
                f"{result['predicted_probability']:.6f}",
                "正确" if result['is_correct'] else "错误"
            ]

            # 添加每个类别的概率
            all_probs = result['all_probabilities']
            for i in range(len(class_names)):
                row.append(f"{all_probs[i]:.6f}")

            f.write("\t".join(row) + "\n")

    print(f"概率表格已保存到: {output_file}")
    return output_file


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load image folder
    folder_path = r"D:\PythonProject6\ResNet\wheat_data\val"  # Update this to your actual validation folder path
    assert os.path.exists(folder_path), "folder: '{}' does not exist.".format(folder_path)

    # Read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    print(f"加载了 {len(class_indict)} 个类别:")
    for idx, name in class_indict.items():
        print(f"  索引 {idx}: {name}")

    # Create model
    model = resnet50(num_classes=len(class_indict)).to(device)

    # Load model weights
    weights_path = "./resNet50.pth"
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print("模型加载成功")

    # Predict all images in the folder and calculate accuracy
    results, accuracy = predict_folder(folder_path, model, device, data_transform, class_indict, true_labels=None)

    # 保存四列格式结果
    four_columns_file = save_four_column_format(results, class_indict)

    # 保存概率表格
    probability_table_file = save_probability_table(results, class_indict)

    # 在控制台显示部分结果
    print("\n" + "=" * 80)
    print("预测结果摘要 (前10个):")
    print("=" * 80)
    print("真实类别\t预测类别\t预测概率\t是否正确")
    print("-" * 80)

    for i, result in enumerate(results[:10]):
        true_class = result['true_class']
        predicted_class = result['predicted_class']
        probability = f"{result['predicted_probability']:.3f}"
        is_correct = "✓" if result['is_correct'] else "✗"

        print(f"{true_class}\t{predicted_class}\t{probability}\t{is_correct}")

    if len(results) > 10:
        print(f"... 还有 {len(results) - 10} 个结果")

    print(f"\n总体准确率: {accuracy:.2%}")

    # 显示各类别准确率
    print("\n各类别准确率:")
    for class_name in class_indict.values():
        class_results = [r for r in results if r['true_class'] == class_name]
        if class_results:
            class_correct = sum(1 for r in class_results if r['is_correct'])
            class_accuracy = class_correct / len(class_results)
            print(f"  {class_name}: {class_accuracy:.2%} ({class_correct}/{len(class_results)})")

    print(f"\n结果文件:")
    print(f"  四列格式: {four_columns_file}")
    print(f"  概率表格: {probability_table_file}")


if __name__ == '__main__':
    main()