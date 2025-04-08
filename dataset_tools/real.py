# 该脚本用于从 DomainNet 数据集的 real.zip 文件中提取指定类别的图片，并保存为扁平化的目录结构

import zipfile
import os
import zipfile
import random 
def get_domainnet_real_classes_random(zip_path, num_classes=10):
    """
    列出 DomainNet real.zip 中包含的所有类别（一级目录名）
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        all_files = z.namelist()
        class_names = set()
        for f in all_files:
            if f.startswith("real/") and len(f.split("/")) >= 3:
                cls = f.split("/")[1]
                class_names.add(cls)
        return random.sample(sorted(class_names),num_classes)


def extract_domainnet_real_subset(
    zip_path,
    dest_dir="./data/real/",
    selected_classes=("cat", "dog", "car")
):
    with zipfile.ZipFile(zip_path, 'r') as z:
        all_files = z.namelist()
        print(f"ZIP 包总文件数：{len(all_files)}")

        for cls in selected_classes:
            class_prefix = f"real/{cls}/"
            class_files = [f for f in all_files if f.startswith(class_prefix) and f.lower().endswith((".jpg", ".jpeg", ".png"))]

            target_class_dir = os.path.join(dest_dir, cls)
            os.makedirs(target_class_dir, exist_ok=True)

            print(f" 正在提取类别 '{cls}' 共 {len(class_files)} 张图片...")

            for file in class_files:
                z.extract(file, path=dest_dir)

            # 移动到扁平结构（real/cat/img.jpg → domainnet_real_subset/cat/img.jpg）
            for file in class_files:
                src = os.path.join(dest_dir, file)
                dst = os.path.join(dest_dir, cls, os.path.basename(file))
                os.rename(src, dst)

        # 清理多余空文件夹（real/）
        real_dir = os.path.join(dest_dir, "real")
        if os.path.exists(real_dir):
            import shutil
            shutil.rmtree(real_dir)

    print(f"\n提取完成！数据保存在：{dest_dir}")
    print("✔ 目录结构：")
    for cls in selected_classes:
        print(f" - {os.path.join(dest_dir, cls)}")


get_classes = get_domainnet_real_classes_random("./data/real.zip")
extract_domainnet_real_subset(
    zip_path="./data/real.zip",
    dest_dir="./data/real/",
    selected_classes=get_classes
)