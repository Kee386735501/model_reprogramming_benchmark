import os
import random
import torch
from torchvision import datasets, transforms

def select_classes_from_imagefolder(root_dir, transform, num_classes=30, seed=42):
    """
    从 ImageFolder 格式的数据集中随机选择若干类别，并返回新的 dataset 对象。

    参数:
        root_dir (str): 原始数据集路径（ImageFolder 格式）
        transform: 应用于图像的 torchvision.transforms
        num_classes (int): 要随机选择的类别数量
        seed (int): 随机种子，确保可复现

    返回:
        new_dataset (ImageFolder): 仅包含选中类别的子集
        selected_classes (list[str]): 被选中的类别名
    """
    # 固定随机种子以保证实验可复现
    random.seed(seed)

    # 加载原始完整数据集
    full_dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    all_classes = full_dataset.classes

    # 检查类别数是否足够
    assert num_classes <= len(all_classes), f"最多可选 {len(all_classes)} 类"

    # 随机选择类别
    selected_classes = random.sample(all_classes, num_classes)
    selected_class_indices = [full_dataset.class_to_idx[c] for c in selected_classes]

    # 过滤出选中类别的样本
    selected_samples = [s for s in full_dataset.samples if s[1] in selected_class_indices]

    # 映射旧标签 → 新标签
    old_to_new = {old: new for new, old in enumerate(selected_class_indices)}
    selected_samples = [(path, old_to_new[label]) for path, label in selected_samples]

    # 构造新的 dataset 对象
    new_dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    new_dataset.samples = selected_samples
    new_dataset.targets = [label for _, label in selected_samples]
    new_dataset.classes = selected_classes
    new_dataset.class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}

    return new_dataset, selected_classes