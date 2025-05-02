
import torch.nn as nn
import torch

import torch
import torch.nn.functional as F
import os
from functools import partial
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from tqdm.auto import tqdm  
from torch.utils.tensorboard import SummaryWriter
import csv
import os
import matplotlib.pyplot as plt
import time 
from torchvision.datasets import SVHN
import random
from torch.utils.data import Subset, DataLoader



# linear probe method
def linear_probe(train_dataset, test_dataset, base_model):
    args = Args()
    device = args.device
    # 冻结参数
    for parameter in base_model.parameters():
        parameter.requires_grad = False
    # 线性分类器(resnet or vit)
    if hasattr(base_model,"fc"):
        feature_dim = base_model.fc.in_features
        base_model.fc = nn.Linear(feature_dim, args.num_classes)
    elif hasattr(base_model,"head"):
        feature_dim = base_model.heads.head.in_features
        base_model.heads = nn.Linear(feature_dim, args.num_classes)
    else:
        raise ValueError("Model does not have a linear layer to replace")
    base_model = base_model.to(device)
    # 优化器 & 损失函数
    optimizer = torch.optim.Adam(base_model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 日志记录
    writer = SummaryWriter(log_dir=f"./logs/linear_probe_{args.model}")
    best_acc = 0

    print("Start linear probe training")
    for epoch in range(args.epochs):
        base_model.train()
        correct = 0
        total = 0
        train_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = base_model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        writer.add_scalar("train/loss", train_loss / total, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)

        # 验证
        base_model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = base_model(x)
                loss = criterion(logits, y)
                test_loss += loss.item() * x.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        test_acc = correct / total
        writer.add_scalar("test/loss", test_loss / total, epoch)
        writer.add_scalar("test/acc", test_acc, epoch)

        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

        best_acc = max(best_acc, test_acc)

    print(f"Best Test Accuracy: {best_acc:.4f}")
    writer.close()





def get_partial_dataset(dataset, fraction=1.0, seed=42):
    """
    Args:
        dataset: 原始完整的数据集（比如 CIFAR10 的 train_dataset）
        fraction: 取多少比例（比如0.5表示用50%的数据）
        seed: 随机种子，保证可复现性
    Returns:
        Subset对象，只包含部分数据
    """
    if fraction >= 1.0:
        return dataset  # 如果fraction>=1，就直接返回原dataset

    # 计算要取的样本数
    num_samples = int(len(dataset) * fraction)

    # 随机采样indices
    random.seed(seed)
    indices = random.sample(range(len(dataset)), num_samples)

    # 用 Subset 包装
    partial_dataset = Subset(dataset, indices)

    return partial_dataset






# network for patch-wise mask
# 该网络用于生成图像的掩码，掩码的大小和形状与输入图像相同
class AttributeNet(nn.Module):
    def __init__(self, layers=5, patch_size=8, channels=3):
        super(AttributeNet, self).__init__()
        self.layers = layers
        self.patch_size = patch_size
        self.channels = channels

        self.pooling = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1) # in_channels , out_channels , kernel_size,stride,padding
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        if self.layers == 5 and self.channels == 3:
            self.conv6 = nn.Conv2d(64, 3, 3, 1, 1)
        elif self.layers == 6:
            self.conv5 = nn.Conv2d(64, 128, 3, 1, 1)
            self.bn5 = nn.BatchNorm2d(128)
            self.relu5 = nn.ReLU(inplace=True)

            if self.channels == 3:
                self.conv6 = nn.Conv2d(128, 3, 3, 1, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        if self.patch_size in [2, 4, 8, 16, 32]:
            y = self.pooling(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        if self.patch_size in [4, 8, 16, 32]:
            y = self.pooling(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = self.relu3(y)
        if self.patch_size in [8, 16, 32]:
            y = self.pooling(y)
        y = self.conv4(y)
        y = self.bn4(y)
        y = self.relu4(y)
        if self.patch_size in [16, 32]:
            y = self.pooling(y)
        if self.layers == 6:
            y = self.conv5(y)
            y = self.bn5(y)
            y = self.relu5(y)
            if self.patch_size == 32:
                y = self.pooling(y)

        if self.channels == 3:
            y = self.conv6(y)
        elif self.channels == 1:
            y = torch.mean(y, dim=1)
        return y


# network for mask training 把 patchwise 扩展成 图片的size 然后合并原图
# 该网络用于训练掩码，掩码的大小和形状与输入图像相同
class InstancewiseVisualPrompt(nn.Module):
    def __init__(self, size, layers=5, patch_size=8, channels=3):
        '''
        Args:
            size: input image size
            layers: the number of layers of mask-training CNN
            patch_size: the size of patches with the same mask value
            channels: 3 means that the mask value for RGB channels are different, 1 means the same
            keep_watermark: whether to keep the reprogram (\delta) in the model
        '''
        super(InstancewiseVisualPrompt, self).__init__()
        if layers not in [5, 6]:
            raise ValueError("Input layer number is not supported")
        if patch_size not in [1, 2, 4, 8, 16, 32]:
            raise ValueError("Input patch size is not supported")
        if channels not in [1, 3]:
            raise ValueError("Input channel number is not supported")
        if patch_size == 32 and layers != 6:
            raise ValueError("Input layer number and patch size are conflict with each other")

        # Set the attribute mask CNN
        self.patch_num = int(size / patch_size)
        self.imagesize = size
        self.patch_size = patch_size
        self.channels = channels
        self.priority = AttributeNet(layers, patch_size, channels)

        # Set reprogram (\delta) according to the image size
        self.size = size
        self.program = torch.nn.Parameter(data=torch.zeros(3, size, size))

    def forward(self, x):
        # patch-wise interpolation 
        attention = self.priority(x).view(-1, self.channels, self.patch_num * self.patch_num, 1).expand(-1, 3, -1, self.patch_size * self.patch_size).view(-1, 3, self.patch_num, self.patch_num, self.patch_size, self.patch_size).transpose(3, 4)
        attention = attention.reshape(-1, 3, self.imagesize, self.imagesize)
        x = x + self.program * attention
        return x




from torch.nn.functional import one_hot


def label_mapping_base(logits, mapping_sequence):
    modified_logits = logits[:, mapping_sequence]
    return modified_logits

def get_dist_matrix(fx, y):
    fx = one_hot(torch.argmax(fx, dim = -1), num_classes=fx.size(-1))
    dist_matrix = [fx[y==i].sum(0).unsqueeze(1) for i in range(len(y.unique()))]
    dist_matrix = torch.cat(dist_matrix, dim=1)
    return dist_matrix

def predictive_distribution_based_multi_label_mapping(dist_matrix, mlm_num: int):
    assert mlm_num * dist_matrix.size(1) <= dist_matrix.size(0), "source label number not enough for mapping"
    mapping_matrix = torch.zeros_like(dist_matrix, dtype=int)
    dist_matrix_flat = dist_matrix.flatten()
    for _ in range(mlm_num * dist_matrix.size(1)):
        loc = dist_matrix_flat.argmax().item()
        loc = [loc // dist_matrix.size(1), loc % dist_matrix.size(1)]
        mapping_matrix[loc[0], loc[1]] = 1
        dist_matrix[loc[0]] = -1
        if mapping_matrix[:, loc[1]].sum() == mlm_num:
            dist_matrix[:, loc[1]] = -1
    return mapping_matrix


def generate_label_mapping_by_frequency(visual_prompt, network, data_loader, mapping_num = 1): # mapping_num=1: 1V1 match
    device = next(visual_prompt.parameters()).device
    print(device)
    if hasattr(network, "eval"):
        network.eval()
    fx0s = []
    ys = []
    dataset = data_loader.dataset  # 获取原始数据集
    print(f"Dataset length: {len(dataset)}")
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Frequency Label Mapping", ncols=100) if len(data_loader) > 20 else data_loader
    # print(f"DataLoader has {len(data_loader)} batches")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            # 
            fx0 = network(visual_prompt(x))
        fx0s.append(fx0)
        ys.append(y)
    fx0s = torch.cat(fx0s).cpu().float()
    ys = torch.cat(ys).cpu().int()
    if ys.size(0) != fx0s.size(0):
        assert fx0s.size(0) % ys.size(0) == 0
        ys = ys.repeat(int(fx0s.size(0) / ys.size(0)))
    dist_matrix = get_dist_matrix(fx0s, ys)
    pairs = torch.nonzero(predictive_distribution_based_multi_label_mapping(dist_matrix, mapping_num)) # (C, C) 原来i类对应现在的j类, j=0,1,...,C
    mapping_sequence = pairs[:, 0][torch.sort(pairs[:, 1]).indices.tolist()]
    return mapping_sequence


def label_mapping_base(logits, mapping_sequence):
    modified_logits = logits[:, mapping_sequence]
    return modified_logits


import torch

class Args:
    def __init__(self):
        self.source = "svhn"
        self.target = "svhn"
        self.mr_ratio = 1.0  # or other default
        self.num_classes = 100
        self.epochs = 50
        self.lr = 0.01
        self.batch_size = 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.whether_pretrained = True
        self.whether_save = False
        self.save_name = None
        self.seed = 42
        self.patch_size = 8
        self.attribute_channels = 3 
        self.attribute_layers = 5
        self.fraction = 1.0 
        self.local_import_model = True
        self.mapping_method = "rlm"
        self.imgsize = 224
        self.attr_gamma = 0.1
        self.attr_lr = 0.01
        self.model = "ViT_B32"  # or other default





# model reprogramming method
from functools import partial
from torch.amp import GradScaler
def reprogram_model(train_dataset,test_dataset,base_model):
    args = Args()
    device = args.device
    if args.model == "ViT_B32":
        args.imgsize = 384
        

        args.imgsize = 224
    model = base_model
    # load the dataset 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # initialize the visual prompting model
    base_model.requires_grad_(False)
    base_model.eval()
    vp = visual_prompt = InstancewiseVisualPrompt(args.imgsize, args.attribute_layers, args.patch_size, args.attribute_channels).to(device)
    optimizer = torch.optim.Adam([{'params': visual_prompt.program, 'lr': args.lr}])
    optimizer_att = torch.optim.Adam([{'params': visual_prompt.priority.parameters(), 'lr': args.attr_lr}])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[int(0.5 * args.epochs), int(0.72 * args.epochs)],
                                                     gamma=0.1)
    scheduler_att = torch.optim.lr_scheduler.MultiStepLR(optimizer_att,
                                                    milestones=[int(0.5 * args.epochs), int(0.72 * args.epochs)],
                                                    gamma=args.attr_gamma)
    class_names = [str(i) for i in range(args.num_classes)]
    if args.mapping_method == 'rlm':
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, args.imgsize, args.imgsize).to(device)
            output_dim = base_model(dummy_input).shape[1]

        mapping_sequence = torch.randperm(output_dim)[:len(class_names)]
        label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
    elif args.mapping_method == 'flm':
        mapping_sequence = generate_label_mapping_by_frequency(visual_prompt, base_model, train_loader)
        label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
    # Train
    train_accs = []
    train_losses = []
    test_accs = []


    best_acc = 0.
    tr_acc = 0
    scaler = GradScaler()
    print("start training")

    # Tensorboard Write in
    # now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir_smm = f"./logs/{args.model}_{args.source}_to_{args.target}_{args.mapping_method}_mr{args.mr_ratio}"
    os.makedirs(log_dir_smm, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir_smm)


    # train loop
    for epoch in range(args.epochs):
        start_time = time.time()
        if args.mapping_method == 'ilm':
            mapping_sequence = generate_label_mapping_by_frequency(visual_prompt, base_model, train_loader)
            label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
        visual_prompt.train()
        total_num = 0
        true_num = 0
        loss_sum = 0
        print(f"Epoch {epoch+1}/{args.epochs} training")
        pbar = tqdm(train_loader, total=len(train_loader),
                    desc=f"Epo {epoch}", ncols=100)
        for x, y in pbar:
            if x.get_device() == -1:
                x, y = x.to(device), y.to(device)
            pbar.set_description_str(f"Epo {epoch}", refresh=True)
            optimizer.zero_grad()
            optimizer_att.zero_grad()
            with autocast():
                fx = label_mapping(base_model(visual_prompt(x)))
                loss = F.cross_entropy(fx, y, reduction='mean')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.step(optimizer_att)
            scaler.update()
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            loss_sum += loss.item() * fx.size(0)
            pbar.set_postfix_str(f"Acc {100 * true_num / total_num:.2f}%")
        epoch_time = time.time() - start_time
        scheduler.step()
        scheduler_att.step()
        
        train_accs.append(true_num / total_num)
        tr_acc = true_num / total_num
        train_losses.append(loss_sum / total_num)
        print("train/acc", true_num / total_num, epoch)
        print("train/loss", loss_sum / total_num, epoch)
        writer.add_scalar("train/acc", true_num / total_num, epoch)
        writer.add_scalar("train/loss", loss_sum / total_num, epoch)
        writer.add_scalar("train/time", epoch_time, epoch)


        # Test
        visual_prompt.eval()
        total_num = 0
        true_num = 0
        loss_sum = 0
        pbar = tqdm(test_loader, total=len(test_loader), desc=f"Epo {epoch} Testing", ncols=100)
        ys = []
        for x, y in pbar:
            if x.get_device() == -1:
                x, y = x.to(device), y.to(device)
            ys.append(y)
            with torch.no_grad():
                fx0 = base_model(visual_prompt(x))
                fx = label_mapping(fx0)
                loss = F.cross_entropy(fx, y, reduction='mean')
            loss_sum += loss.item() * fx.size(0)
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            acc = true_num / total_num
            pbar.set_postfix_str(f"Acc {100 * acc:.2f}%")
        print("test/acc", acc, epoch)
        print("test/loss", loss_sum / total_num, epoch)
        writer.add_scalar("test/acc", acc, epoch)
        writer.add_scalar("test/loss", loss_sum / total_num, epoch)

        # save data 
        train_acc = tr_acc
        train_loss = loss_sum / total_num
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        # ====== 写入日志文件 ======
        writer.flush()
        writer.close()


# noise part
from PIL import Image, ImageFilter
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

import torch
from torch.utils.data import Dataset

class FastNoisyDataset(Dataset):
    def __init__(self, base_dataset, noise_type="gaussian", noise_level=0.1):
        self.base_dataset = base_dataset
        self.noise_type = noise_type
        self.noise_level = noise_level

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]  # img 已经是 Tensor 格式 (C, H, W)

        if self.noise_level > 0:
            if self.noise_type == "gaussian":
                noise = torch.randn_like(img) * self.noise_level
                img = torch.clamp(img + noise, 0.0, 1.0)

            elif self.noise_type == "salt_pepper":
                prob = self.noise_level
                rand = torch.rand_like(img)
                img = torch.where(rand < prob / 2, torch.zeros_like(img), img)  # salt
                img = torch.where(rand > 1 - prob / 2, torch.ones_like(img), img)  # pepper

            elif self.noise_type == "blur":
                from torchvision.transforms.functional import gaussian_blur
                img = gaussian_blur(img, kernel_size=3)

        return img, label

from torchvision.datasets import CIFAR100, SVHN

def get_dataset(name,
                root: str = './data',
                image_size: int = 224,
                download: bool = True):

    if name == "cifar100":
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],  # CIFAR-100 官方统计
                                 std=[0.2675, 0.2565, 0.2761])
        ])
        train_dataset = CIFAR100(root=root, train=True, transform=transform, download=download)
        test_dataset = CIFAR100(root=root, train=False, transform=transform, download=download)
        num_classes = 100

    elif name == "svhn":
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],  # SVHN 官方统计
                                 std=[0.1980, 0.2010, 0.1970])
        ])
        train_dataset = SVHN(root=root, split='train', transform=transform, download=download)
        test_dataset = SVHN(root=root, split='test', transform=transform, download=download)
        num_classes = 10

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return train_dataset, test_dataset, num_classes


def get_noisy_dataset(base_dataset: Dataset, noise_type: str = "gaussian", noise_level: float = 0.1) -> Dataset:
    return FastNoisyDataset(base_dataset, noise_type=noise_type, noise_level=noise_level)


def get_model(name: str, pretrained: bool = True):
    if name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        # model.fc = nn.Identity()  # 去掉最后的全连接层
    elif name == "vit_b_16":
        from torchvision.models import vit_b_16,ViT_B_16_Weights
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        model = vit_b_16(weights=weights)
        # model.heads = nn.Linear(768,100)
        image_size = 224
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model, image_size

# prepare dataset 

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize(224),  # 适配 ResNet18 预训练的输入
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 的 mean 和 std
                         std=[0.229, 0.224, 0.225]),
])


# svhn_test_dataset = SVHN(root='./data', split='test', download=True, transform=transform)
# svhn_transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])  # SVHN 统计
# ])



# train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=12, prefetch_factor=2, persistent_workers=True)
# test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=12, prefetch_factor=2, persistent_workers=True)

train_dataset,test_dataset,num_classes = get_dataset(name="cifar100", root="./data", image_size=224, download=True)
# add noise to dataset
noisy_train_dataset = get_noisy_dataset(train_dataset, noise_type="salt_pepper", noise_level=0.1)
noisy_test_dataset = get_noisy_dataset(test_dataset, noise_type="salt_pepper", noise_level=0.1)

# new DataLoader
noise_train_loader = DataLoader(noisy_train_dataset, batch_size=512, shuffle=True, num_workers=4)
noise_test_loader = DataLoader(noisy_test_dataset, batch_size=512, shuffle=False, num_workers=4)


# reprogram model
import torchvision.models as models
import torch

# backbone = models.resnet18(pretrained=True)
# backbone.fc = torch.nn.Identity()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)  # 应该输出 cuda:0

base_model,image_size = get_model("vit_b_16", pretrained=True)
base_model = base_model.to(device)
reprogram_model(
 train_dataset=noisy_train_dataset,
    test_dataset=noisy_test_dataset,
    base_model=base_model.to(device),
)
