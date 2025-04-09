# config.py

import torch

class Args:
    def __init__(self):
        self.num_classes = 10
        self.epochs = 100
        self.lr = 0.001
        self.batch_size = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.whether_pretrained = False
        self.whether_save = False
        self.save_name = None
        self.seed = 42
        self.model = "resnet18"
        self.patch_size = 8
        self.attribute_channels = 3 
        self.attribute_layers = 5
        self.fraction = 1.0 
        self.local_import_model = True
        self.mapping_method = "rlm"
        self.imgsize = 224
        self.attr_gamma = 0.1
        self.attr_lr = 0.01
