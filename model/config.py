import os
import numpy as np
import requests
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms

class input_dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label



import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_image_dataset(data_dir, image_size=(224, 224), batch_size=32, shuffle=True):
    """
    加载文件夹结构的图像分类数据集

    参数:
        data_dir (str): 数据集根目录（子文件夹为类别名）
        image_size (tuple): 图像缩放大小，默认 (224, 224)
        batch_size (int): 每个 batch 的大小
        shuffle (bool): 是否打乱数据顺序

    返回:
        dataloader: PyTorch DataLoader 对象
        class_names: 类别名称列表
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader, dataset.classes

















def load_custom_dataset(data, labels, samples_per_class=None, test_size=0.2, input_size=(224, 224)):
    """
    加载自定义数据集并划分为训练集和测试集，同时可以控制每个类别的数据数量。

    参数:
        data (numpy.ndarray): 输入数据，形状为 (N, H, W) 或 (N, H, W, C)。
        labels (numpy.ndarray): 对应的标签，形状为 (N,)。
        samples_per_class (int, optional): 每个类别包含的样本数量。如果为 None，则使用所有可用数据。
        test_size (float): 测试集所占数据集的比例。
        input_size (tuple): 模型输入的目标尺寸。

    返回:
        train_dataset (Dataset): 训练数据集。
        test_dataset (Dataset): 测试数据集。
    """
    if samples_per_class is not None:
        selected_data = []
        selected_labels = []
        unique_classes = np.unique(labels)  # 获取所有唯一类别
        for cls in unique_classes:
            cls_indices = np.where(labels == cls)[0]  # 获取当前类别的所有索引
            cls_indices = np.random.choice(cls_indices, samples_per_class, replace=False)  # 随机选择指定数量的样本
            selected_data.append(data[cls_indices])  # 添加选中的数据
            selected_labels.append(labels[cls_indices])  # 添加对应的标签
        data = np.concatenate(selected_data, axis=0)  # 合并所有类别的数据
        labels = np.concatenate(selected_labels, axis=0)  # 合并所有类别的标签

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, stratify=labels, random_state=42
    )

    # 定义数据预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),  # 调整到目标输入尺寸
        transforms.ToTensor(),
    ])

    # 创建训练集和测试集
    train_dataset = input_dataset(X_train, y_train, transform)
    test_dataset = input_dataset(X_test, y_test, transform)
    return train_dataset, test_dataset