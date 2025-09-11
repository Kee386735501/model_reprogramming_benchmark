# models.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

# ===== 黑盒模型加载器 =====
def get_blackbox_model(args):
    """
    加载并冻结预训练的黑盒模型
    """
    if args.model_name == 'resnet18':
        if args.weight_path != 'none' and args.weight_path:
            model = resnet18(weights=None)
            # 确保FC层与权重匹配
            model.fc = nn.Linear(model.fc.in_features, args.input_dim)
            state = torch.load(args.weight_path, map_location='cpu')
            model.load_state_dict(state)
            print(f"[weights] Loaded custom weights from {args.weight_path}")
        else:
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            print("[weights] Loaded ImageNet pre-trained weights for ResNet18.")

    elif args.model_name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        print("[weights] Loaded ImageNet pre-trained weights for ResNet50.")
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    model = model.to(args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

# ===== 固定映射分类器 (用于 Model Reprogramming) =====
class FixedMappingClassifier(nn.Module):
    def __init__(self, blackbox_model, mapping_matrix, visual_prompt, temp=0.6):
        super().__init__()
        self.blackbox = blackbox_model
        self.visual_prompt = visual_prompt
        self.temp = temp
        self.register_buffer("mapping", mapping_matrix.float())

    def forward(self, x):
        x = self.visual_prompt(x)
        fx = self.blackbox(x)
        fx_probs = torch.softmax(fx / self.temp, dim=1)
        return fx_probs @ self.mapping

# ===== 可训练线性分类器 (用于 Linear Probing) =====
class TrainableLinearClassifier(nn.Module):
    def __init__(self, blackbox_model, visual_prompt, input_dim=1000, output_dim=10):
        super().__init__()
        self.blackbox = blackbox_model
        self.linear = nn.Linear(input_dim, output_dim)
        self.visual_prompt = visual_prompt

    def forward(self, x):
        x = self.visual_prompt(x)
        with torch.no_grad(): # 黑盒模型特征提取不计算梯度
            fx = self.blackbox(x)
        return self.linear(fx)
    


# ===== 可训练线性分类器 (用于 F θ) =====

class TrainableFLinear(nn.Module):
    def __init__(self, blackbox_model, output_dim=10):
        super().__init__()
        # 记录输入维度（比如 ResNet18 的 fc.in_features = 512）
        in_features = blackbox_model.fc.in_features
        # 去掉原有分类头
        blackbox_model.fc = nn.Identity()
        self.blackbox = blackbox_model
        # 新的线性头（可训练）
        self.linear = nn.Linear(in_features, output_dim)

    def forward(self, x):
        with torch.no_grad():  # backbone 冻结
            feats = self.blackbox(x)   # 输出特征 [B, in_features]
        return self.linear(feats)      # 分类 logits
