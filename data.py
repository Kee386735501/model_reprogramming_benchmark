from torchvision import transforms
import os
from torch.utils.data import DataLoader
from torchvision import datasets
from const import GTSRB_LABEL_MAP
from torch.utils.data import DataLoader, Subset   # ← 添加 Subset
import torch   

imgsize = 224
padding_size = 4 
IMAGENETNORMALIZE = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}


def _subset_by_percent(ds, percent: int, seed: int):
    if percent >= 100:
        return ds
    n = len(ds)
    m = max(1, int(round(n * percent / 100.0)))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:m].tolist()
    return Subset(ds, idx)




# 获取训练预处理
def get_train_preprocess(vp_mode,padding_px: int = 16):
    norm = transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std'])
    to_rgb = transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x)
    if vp_mode == "watermark":
        train_preprocess = transforms.Compose([
            transforms.Resize((imgsize + 4, imgsize + 4)),
            transforms.RandomCrop(imgsize),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
        ])
        test_preprocess = transforms.Compose([
            transforms.Resize((imgsize, imgsize)),
            transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
        ])
    elif vp_mode == "instance":
        train_preprocess = transforms.Compose([
        transforms.Resize((imgsize + 32, imgsize + 32)),
        transforms.RandomCrop(imgsize),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
    ])
        test_preprocess = transforms.Compose([
            transforms.Resize((imgsize, imgsize)),
            transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
        ])
    elif vp_mode == "padding":
        # 给图像四周加 padding，再随机裁回原尺寸，等效为给 VP 预留边缘空间
        train_preprocess = transforms.Compose([
            transforms.Resize((imgsize, imgsize)),
            transforms.Pad(padding_px, fill=0, padding_mode='constant'),
            transforms.RandomCrop(imgsize),
            transforms.RandomHorizontalFlip(),
            to_rgb, transforms.ToTensor(), norm,
        ])
        test_preprocess = transforms.Compose([
            transforms.Resize((imgsize, imgsize)),
            to_rgb, transforms.ToTensor(), norm,
        ])
    elif vp_mode == "none":
        # 无 VP：标准增广；不要扩边
        train_preprocess = transforms.Compose([
            transforms.Resize((imgsize, imgsize)),
            transforms.RandomHorizontalFlip(),
            to_rgb, transforms.ToTensor(), norm,
        ])
        test_preprocess = transforms.Compose([
            transforms.Resize((imgsize, imgsize)),
            to_rgb, transforms.ToTensor(), norm,
        ])
    else:
        # 兜底：与 none 相同，保证不报错
        train_preprocess = transforms.Compose([
            transforms.Resize((imgsize, imgsize)),
            transforms.RandomHorizontalFlip(),
            to_rgb, transforms.ToTensor(), norm,
        ])
        test_preprocess = transforms.Compose([
            transforms.Resize((imgsize, imgsize)),
            to_rgb, transforms.ToTensor(), norm,
        ])
    return train_preprocess, test_preprocess


# 获取数据集 

def refine_classnames(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ')
    return class_names


def prepare_additive_data(dataset, data_path, preprocess, test_process=None,
                          batch_size=1024, shuffle=True,
                          train_percent=100, seed=42):    # ← 新增参数
    data_path = os.path.join(data_path, dataset)
    if not test_process:
        test_process = preprocess
    if dataset == "cifar10":
        train_data = datasets.CIFAR10(root = data_path, train = True, download = True, transform = preprocess)
        train_data = _subset_by_percent(train_data, train_percent, seed)
        test_data = datasets.CIFAR10(root = data_path, train = False, download = True, transform = test_process)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, batch_size, shuffle = shuffle, num_workers=1),
            'test': DataLoader(test_data, batch_size, shuffle = False, num_workers=1),
        }     
    elif dataset == "cifar100":
        train_data = datasets.CIFAR100(root = data_path, train = True, download = True, transform = preprocess)
        train_data = _subset_by_percent(train_data, train_percent, seed)
        test_data = datasets.CIFAR100(root = data_path, train = False, download = True, transform = test_process)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, batch_size, shuffle = shuffle, num_workers=1),
            'test': DataLoader(test_data, batch_size, shuffle = False, num_workers=1),
        }
    elif dataset == "svhn":
        train_data = datasets.SVHN(root = data_path, split="train", download = True, transform = preprocess)
        train_data = _subset_by_percent(train_data, train_percent, seed)
        test_data = datasets.SVHN(root = data_path, split="test", download = True, transform = test_process)
        class_names = [f'{i}' for i in range(10)]
        loaders = {
            'train': DataLoader(train_data, batch_size, shuffle = shuffle, num_workers=1),
            'test': DataLoader(test_data, batch_size, shuffle = False, num_workers=1),
        }
    elif dataset == "gtsrb":
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        train_data = _subset_by_percent(train_data, train_percent, seed)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = test_process)
        class_names = refine_classnames(list(GTSRB_LABEL_MAP.values()))
        loaders = {
            'train': DataLoader(train_data, batch_size, shuffle = shuffle, num_workers=1),
            'test': DataLoader(test_data, batch_size, shuffle = False, num_workers=1),
        }
    else:
        raise NotImplementedError(f"{dataset} not supported")

    return loaders, class_names