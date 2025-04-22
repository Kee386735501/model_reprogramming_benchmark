import os
import shutil
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from pytorch_fid import fid_score
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader

# ======================== Config ========================
dataset_root = '/data/projects/punim1943/domainnet'
domain1 = 'sketch'
domain2 = 'real'
output_file = 'distance_results.txt'
distance_type = 'mmd'  # 可选：'fid', 'mmd'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ======================== 工具函数 ========================

def is_valid_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg'))

def get_valid_common_classes(path1, path2):
    def valid_class_names(domain_path):
        return {
            name for name in os.listdir(domain_path)
            if os.path.isdir(os.path.join(domain_path, name)) and
            any(is_valid_image_file(f) for f in os.listdir(os.path.join(domain_path, name)))
        }

    classes1 = valid_class_names(path1)
    classes2 = valid_class_names(path2)
    return sorted(list(classes1 & classes2))

def resize_and_copy_images(src_dir, dst_dir, size=(256, 256)):
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        if not is_valid_image_file(fname):
            continue
        try:
            img = Image.open(os.path.join(src_dir, fname)).convert('RGB')
            img = img.resize(size, Image.BICUBIC)
            img.save(os.path.join(dst_dir, fname))
        except Exception as e:
            print(f"Skipping {fname}: {e}")

def compute_fid_for_class(path1, path2, tmp_base='tmp_resized'):
    tmp1 = os.path.join(tmp_base, 'domain1')
    tmp2 = os.path.join(tmp_base, 'domain2')
    if os.path.exists(tmp_base):
        shutil.rmtree(tmp_base)
    resize_and_copy_images(path1, tmp1)
    resize_and_copy_images(path2, tmp2)
    score = fid_score.calculate_fid_given_paths([tmp1, tmp2], batch_size=50, device=device, dims=2048)
    shutil.rmtree(tmp_base)
    return score

def compute_mmd_distance(path1, path2):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    class ImageFolderSimple(Dataset):
        def __init__(self, path):
            self.path = path
            self.files = [f for f in os.listdir(path) if is_valid_image_file(f)]
            self.loader = default_loader
        def __len__(self):
            return len(self.files)
        def __getitem__(self, idx):
            return transform(self.loader(os.path.join(self.path, self.files[idx])))

    loader1 = DataLoader(ImageFolderSimple(path1), batch_size=32, shuffle=False)
    loader2 = DataLoader(ImageFolderSimple(path2), batch_size=32, shuffle=False)

    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()

    def extract_features(loader):
        feats = []
        with torch.no_grad():
            for x in loader:
                x = x.to(device)
                f = model(x)
                feats.append(f.cpu())
        return torch.cat(feats, dim=0)

    feat1 = extract_features(loader1)
    feat2 = extract_features(loader2)

    def gaussian_kernel(x, y, sigma=1.0):
        beta = 1.0 / (2.0 * sigma ** 2)
        dists = torch.cdist(x, y) ** 2
        return torch.exp(-beta * dists)

    kxx = gaussian_kernel(feat1, feat1)
    kyy = gaussian_kernel(feat2, feat2)
    kxy = gaussian_kernel(feat1, feat2)

    mmd = kxx.mean() + kyy.mean() - 2 * kxy.mean()
    return mmd.item()

def compute_global_average(values_dict):
    vals = [v for v in values_dict.values() if isinstance(v, (float, int))]
    return sum(vals) / len(vals) if vals else None

# ======================== 主函数 ========================

def main():
    path1_root = os.path.join(dataset_root, domain1)
    path2_root = os.path.join(dataset_root, domain2)

    if not os.path.exists(path1_root) or not os.path.exists(path2_root):
        print(f"❌ One or both domains do not exist: {path1_root}, {path2_root}")
        return

    common_classes = get_valid_common_classes(path1_root, path2_root)
    print(f"✅ Found {len(common_classes)} common classes: {common_classes}")

    results = {}
    with open(output_file, 'w') as f:
        for cls in tqdm(common_classes, desc=f"Computing {distance_type.upper()}"):
            path1 = os.path.join(path1_root, cls)
            path2 = os.path.join(path2_root, cls)
            try:
                if distance_type == 'fid':
                    score = compute_fid_for_class(path1, path2)
                elif distance_type == 'mmd':
                    score = compute_mmd_distance(path1, path2)
                else:
                    raise ValueError(f"Unsupported distance type: {distance_type}")
                results[cls] = score
                line = f"{cls}: {distance_type.upper()} = {score:.4f}"
            except Exception as e:
                results[cls] = None
                line = f"{cls}: Error: {e}"
            print(line)
            f.write(line + "\n")

        avg = compute_global_average(results)
        if avg is not None:
            f.write(f"\nGlobal Average {distance_type.upper()}: {avg:.4f}\n")
            print(f"\n✅ Global Average {distance_type.upper()}: {avg:.4f}")

if __name__ == '__main__':
    main()
