# config.py
import argparse
from html import parser

class Config:
    def __init__(self, custom_args=None):  # 增加可选参数 custom_args
        parser = argparse.ArgumentParser(description="Visual Prompt Training")

        # 设备选择
        parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

        # VP 相关
        # config.py
        parser.add_argument('--vp_mode', type=str, default='none',
                    choices=['watermark', 'instance', 'padding', 'none'], help="choose the appropriate visual prompt mode,default is no vp")  # ← 加上 padding/none
        parser.add_argument('--mapping_mode', type=str, default='none',
                    choices=['ilm', 'blm', 'blm+','none'], help="choose the appropriate mapping mode")  # ← 全小写

        # 模型选择
        parser.add_argument('--model_name', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
        parser.add_argument('--image_size', type=int, default=224, help="Input image size for the model") 

        # 数据参数
        parser.add_argument('--dataset', type=str, default='cifar10')
        parser.add_argument('--input_dim', type=int, default=1000, help="Number of classes for the top layer")
        parser.add_argument('--data_root', type=str, default='./data')
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--shuffle', type=bool, default=True)
        parser.add_argument('--output_dim', type=int, default=10, help="target dataset dimension")
        parser.add_argument("--epochs",type=int,default=50)
        # 控制训练数据集比例
        parser.add_argument('--train_percent', type=int, default=100,
                    choices=[20, 40, 60, 80, 100],
                    help="Percent of training set to use")
        parser.add_argument('--seed', type=int, default=42, help="Random seed for subset sampling")

        # parser.add_argument('--input_size', type=int, default=4, help="Number of workers for data loading")
        # 导入模型权重
        parser.add_argument('--weight_path', type=str, default='none',
    help='可选：模型权重文件路径（留空则不加载）')

        # 兼容 Jupyter Notebook
        if custom_args is not None:
            self.args = parser.parse_args(custom_args)
        else:
            self.args = parser.parse_args([])  # 避免 Jupyter argv 冲突

    def get(self):
        return self.args
