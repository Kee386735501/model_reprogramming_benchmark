"""
quickdraw.py

该脚本提供了一个函数，用于从Quick, Draw! 数据集中下载指定类别的手绘草图数据，并将其保存为本地文件。

函数:
--------
download_quickdraw_class(class_name, save_dir="./data/quickdraw/"):
    下载指定类别的Quick, Draw! 数据集文件（.npy格式）。
    
    参数:
    - class_name (str): 要下载的类别名称，例如 "cat" 或 "dog"。
    - save_dir (str): 保存下载文件的目录路径，默认为 "./data/quickdraw/"。

    返回:
    - file_path (str): 下载文件的本地路径。

    功能:
    - 如果指定类别的文件尚未下载，则从 Google Cloud Storage 下载。
    - 如果文件已存在，则跳过下载并提示用户。
"""
import os
import requests

def download_quickdraw_class(class_name, save_dir="./data/quickdraw/"):
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{class_name}.npy"
    file_path = os.path.join(save_dir, filename)
    if not os.path.exists(file_path):
        url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{class_name}.npy"
        print(f"Downloading {filename} ...")
        response = requests.get(url)
        with open(file_path, 'wb') as f:
            f.write(response.content)
    else:
        print(f"{filename} already exists.")
    return file_path


