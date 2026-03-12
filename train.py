import os
import sys
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from model import resnet50


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 直接使用当前目录作为数据集路径
    image_path = r"D:\PythonProject6\ResNet\wheat_photos"

    print(f"使用数据集路径: {image_path}")

    # 检查必要的文件夹
    train_path = os.path.join(image_path, "train")
    val_path = os.path.join(image_path, "val")
    test_path = os.path.join(image_path, "test")

    if not os.path.exists(train_path):
        print(f"错误: 训练文件夹不存在: {train_path}")
        return
    if not os.path.exists(val_path):
        print(f"错误: 验证文件夹不存在: {val_path}")
        return
    if not os.path.exists(test_path):
        print(f"错误: 测试文件夹不存在: {test_path}")
        return

    # 加载数据集
    train_dataset = datasets.ImageFolder(root=train_path,
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # 获取类别映射
    animal_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in animal_list.items())

    print(f"找到 {len(cla_dict)} 个类别: {list(cla_dict.values())}")

    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)

    validate_dataset = datasets.ImageFolder(root=val_path,
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    test_dataset = datasets.ImageFolder(root=test_path,
                                        transform=data_transform["test"])
    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=nw)

    print("using {} images for training, {} images for validation, {} images for testing.".format(train_num,
                                                                                                  val_num,
                                                                                                  test_num))

    # 修改 resnet50 的调用方式 - 使用实际的类别数量
    num_classes = len(cla_dict)
    net = resnet50(num_classes=num_classes)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer - 不在训练时使用正则化
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.01)  # 不使用weight_decay

    epochs = 50
    best_acc = 0.0
    save_path = './resNet50.pth'

    train_steps = len(train_loader)

    # 打开日志文件
    log_file = open(r"E:\traindata100\1.txt", 'w')
    log_file.write(r"Epoch\tTrain Loss\tTrain Accuracy\tVal Loss\tVal Accuracy\tTest Loss\tTest Accuracy\n")

    for epoch in range(epochs):
        # train - 不使用任何正则化
        net.train()
        running_loss = 0.0
        train_acc = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # 统计损失和准确率
            running_loss += loss.item()
            predict_y = torch.max(logits, dim=1)[1]
            train_acc += torch.eq(predict_y, labels.to(device)).sum().item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # 计算平均训练损失和准确率
        avg_train_loss = running_loss / train_steps
        avg_train_acc = train_acc / train_num

        # validate
        net.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                val_loss += loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                val_acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        # 计算平均验证损失和准确率
        avg_val_loss = val_loss / len(validate_loader)
        avg_val_acc = val_acc / val_num

        # test
        test_loss = 0.0
        test_acc = 0.0
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for test_data in test_bar:
                test_images, test_labels = test_data
                outputs = net(test_images.to(device))
                loss = loss_function(outputs, test_labels.to(device))
                test_loss += loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                test_acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

        # 计算平均测试损失和准确率
        avg_test_loss = test_loss / len(test_loader)
        avg_test_acc = test_acc / test_num

        print(
            '[epoch %d] train_loss: %.3f  train_acc: %.3f  val_loss: %.3f  val_acc: %.3f  test_loss: %.3f  test_acc: %.3f' %
            (epoch + 1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc, avg_test_loss, avg_test_acc))

        # 写入日志文件
        log_file.write("{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(
            epoch + 1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc, avg_test_loss, avg_test_acc))

        # 保存当前最佳模型（基于验证集性能）
        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(net.state_dict(), save_path)
            print(f"✅ 保存最佳模型，验证准确率: {best_acc:.4f}")

    log_file.close()  # 关闭日志文件

    print(f"Best validation accuracy: {best_acc:.4f}")
    print('Finished Training')


if __name__ == '__main__':
    main()