!pip install pytorch-fid tqdm
import shutil

import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pytorch_fid import fid_score

# 设置你要比较的两个域（domain）
dataset_root = '/data/projects/punim1943/domainnet'
domain1 = 'real'
domain2 = 'clipart'
output_file = 'fid_results.txt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def is_valid_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg'))

# 返回两个域中公共的有效类别（都存在且包含图像）
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
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
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

    # 清空临时文件夹
    if os.path.exists(tmp_base):
        shutil.rmtree(tmp_base)

    resize_and_copy_images(path1, tmp1)
    resize_and_copy_images(path2, tmp2)

    score = fid_score.calculate_fid_given_paths(
        [tmp1, tmp2],
        batch_size=50,
        device=device,
        dims=2048
    )

    # 清理临时目录
    shutil.rmtree(tmp_base)
    return score

# 计算所有有效 FID 的平均值
def compute_global_average_fid(results_dict):
    fid_values = [v for v in results_dict.values() if isinstance(v, (float, int))]
    if not fid_values:
        return None
    avg = sum(fid_values) / len(fid_values)
    return avg


def main():
    path1_root = os.path.join(dataset_root, domain1)
    path2_root = os.path.join(dataset_root, domain2)

    if not os.path.exists(path1_root) or not os.path.exists(path2_root):
        print(f"❌ One or both domains do not exist: {path1_root}, {path2_root}")
        return

    common_classes = get_valid_common_classes(path1_root, path2_root)
    print(f"✅ Found {len(common_classes)} common classes: {common_classes}")

    with open(output_file, 'w') as f:
        for cls in tqdm(common_classes):
            path1 = os.path.join(path1_root, cls)
            path2 = os.path.join(path2_root, cls)
            try:
                fid = compute_fid_for_class(path1, path2)
                result = f"{cls}: FID = {fid:.4f}"
            except Exception as e:
                result = f"{cls}: Error: {str(e)}"
            print(result)
            f.write(result + "\n")

if __name__ == '__main__':
    main()
