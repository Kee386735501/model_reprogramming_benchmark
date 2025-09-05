import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18,resnet50
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm
from vp import build_visual_prompt,DummyVisualPrompt  
from mapping import one2one_mappnig_matrix, blm_reweight_matrix, blmp_reweight_matrix
import copy 
from data import get_train_preprocess, IMAGENETNORMALIZE, prepare_additive_data
from config import Config
from torch.cuda.amp import GradScaler
import os
from contextlib import nullcontext


# 支持 CUDA 和 MPS，并在都不可用时回退 CPU
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEFAULT_DEVICE = pick_device()
print(f"[DEV] base device: {DEFAULT_DEVICE} | cuda={torch.cuda.is_available()} | "
      f"mps={getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()}")


# 获取 AMP 上下文    
def get_amp_ctx(device):
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            return torch.amp.autocast(device_type="cuda")  # PyTorch 2.x
        except TypeError:
            return torch.cuda.amp.autocast()               # 兼容旧版本
    else:
        return nullcontext()



# ===== Process Input =====
def process_input(args):
    train_transform,test_transform = get_train_preprocess(args.vp_mode)  # 获取训练预处理
    # train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    loaders, class_names = prepare_additive_data(
    dataset=args.dataset,
    data_path=args.data_root,
    preprocess=train_transform,
    test_process=test_transform,
    batch_size=args.batch_size,
    shuffle=args.shuffle,
    train_percent=args.train_percent,   # ← 新增
    seed=getattr(args, "seed", 42)                      # ← 新增
)

    train_loader = loaders['train']
    test_loader = loaders['test']
    return train_loader, test_loader, class_names


# ===== Black-box model (frozen pretrained ResNet18) =====
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

def get_blackbox_model(args):
    if args.model_name == 'resnet18':
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif args.model_name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    model = model.to(args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

# ===== Fixed Mapping Classifier =====
class FixedMappingClassifier(nn.Module):
    def __init__(self, blackbox_model, mapping_matrix, visual_prompt, temp=0.6):
        super().__init__()
        self.blackbox = blackbox_model
        self.visual_prompt = visual_prompt
        self.temp = temp
        self.register_buffer("mapping", mapping_matrix.float())

    @torch.no_grad()
    def update_mapping(self, new_mapping: torch.Tensor):
        self.mapping = new_mapping.to(self.mapping.device).float()

    def forward(self, x):
        x = self.visual_prompt(x)
        fx = self.blackbox(x)                          # [B, P]
        fx = torch.softmax(fx / self.temp, dim=1)      # [B, P]
        return fx @ self.mapping                       # [B, C]
    
    
# ===== 选择映射方式 =====
@torch.no_grad()
def _build_mapping_matrix(mapping_mode,
                          visual_prompt,
                          network,
                          train_loader,
                          device,
                          pretrain_num_classes,
                          target_num_classes,
                          lap_ilm=None,
                          lap_blm=1,
                          lap_blmp=2,
                          topk_ratio=0.02):
    """
    根据 mapping_mode 计算 [P, C] 的映射矩阵（P=预训练类数, C=目标类数），放到 device 上。
    """
    if mapping_mode == 'ilm':
        # ILM 通常返回每个目标类在预训练空间中的索引；这里统一转成矩阵
        seq = one2one_mappnig_matrix(visual_prompt, network, train_loader)  # 长度应为 C
        # 将索引序列转为 one-hot 矩阵 [P, C]
        mapping = torch.zeros(pretrain_num_classes, target_num_classes, device=device)
        mapping[seq, torch.arange(target_num_classes, device=device)] = 1.0
        return mapping.float()

    elif mapping_mode == 'blm':
        mapping = blm_reweight_matrix(visual_prompt, network, train_loader, lap=lap_blm)
        return mapping.to(device).float()

    elif mapping_mode in ('blmp', 'blm+'):
        # Top-k 建议在「预训练类空间」截断
        k = max(1, int(pretrain_num_classes * topk_ratio))
        mapping = blmp_reweight_matrix(visual_prompt, network, train_loader, lap=lap_blmp, k=k)
        return mapping.to(device).float()

    else:
        raise ValueError(f"Unknown mapping mode: {mapping_mode}")


# 不训练直接计算一次矩阵然后评估结果
def eval_zero_shot_mapping(blackbox_model,
                           loaders,              # {'train':..., 'test':...} 或 {'train':..., 'val':...}
                           class_names,
                           mapping_mode,
                           device="cuda",
                           temp=0.6,
                           lap_blm=1,
                           lap_blmp=2,
                           topk_ratio=0.02,
                           renorm_columns=False):
    """
    Zero-shot：不训练任何参数。
    - 用 loaders['train'] 估计一次映射矩阵
    - 用固定映射在 loaders['test']/['val'] 上评测
    返回: (acc, mapping_matrix)
    """
    # 取出 dataloader（兼容 test/val 命名）
    assert 'train' in loaders, "loaders 必须包含 'train'"
    train_loader = loaders['train']
    test_loader = loaders.get('test', loaders.get('val'))
    assert test_loader is not None, "loaders 需要包含 'test' 或 'val'"

    # 规范化映射名
    mm = mapping_mode.lower()
    if mm == "blm+":
        mm = "blmp"

    # 无 VP：恒等
    visual_prompt = DummyVisualPrompt().to(device)

    # 预训练类别数 P、目标类别数 C
    P = getattr(getattr(blackbox_model, "fc", None), "out_features", 1000)
    C = len(class_names)

    # === 估计一次映射矩阵（仅用训练集） ===
    mapping = _build_mapping_matrix(
        mapping_mode=mm,
        visual_prompt=visual_prompt,
        network=blackbox_model,
        train_loader=train_loader,
        device=device,
        pretrain_num_classes=P,
        target_num_classes=C,
        lap_blm=lap_blm,
        lap_blmp=lap_blmp,
        topk_ratio=topk_ratio,
    )  # [P, C]

    if renorm_columns:
        # 可选：把每一列归一化，确保列和为1
        mapping = mapping.clamp_min(0)
        mapping = mapping / (mapping.sum(0, keepdim=True) + 1e-8)

    # === 固定映射做推理并评测 ===
    fixed = FixedMappingClassifier(
        blackbox_model, mapping.to(device), visual_prompt, temp=temp
    ).to(device)

    acc = evaluate(fixed, test_loader, device)
    return acc, mapping





# 训练 mapping+vp
def train_visual_prompt_with_mapping(
    network,                         # 预训练 backbone（冻结参数）
    visual_prompt,                   # 待训练的 prompt 模块（加入 optimizer）
    loaders,                         # {'train': ..., 'test': ...}
    epochs,                          # 总轮数
    optimizer,                       # 只包含 visual_prompt 参数
    scheduler,                       # MultiStepLR 或其他（按 epoch 调用 step）
    device="cuda",
    mapping_mode="blmp",             # 'ilm' | 'blm' | 'blmp'/'blm+'
    class_names=None,                # 目标类名列表（用于取 C）
    pretrain_num_classes=1000,       # P
    temp=0.6,                        # softmax 温度
    lap_blm=1,                       # BLM 的 lap
    lap_blmp=2,                      # BLMP 的 lap
    topk_ratio=0.02,                 # BLMP 的 Top-k 比例（相对 P）
    save_best_path=None,             # 若给路径，则保存最优 visual_prompt.state_dict()
    log_interval=0                   # >0 时每隔若干 step 打印一次 loss
):
    """
    返回: best_acc(float), best_state(dict or None)
    说明:
      - 每个 epoch 先基于 train_loader 估计/更新一次 mapping 矩阵；
      - 训练时只更新 visual_prompt；network 冻结并 eval()；
      - 前向: fx = softmax(network(visual_prompt(x))/temp) @ mapping_matrix；
      - 评测时复用该 epoch 的 mapping。
    """
    assert 'train' in loaders and 'test' in loaders, "loaders 需要包含 'train' 与 'test'"
    assert class_names is not None and len(class_names) > 0, "必须提供目标类名列表以确定目标类别数"

    target_num_classes = len(class_names)
    scaler = GradScaler() if (device.type == "cuda" and torch.cuda.is_available()) else None

    # 冻结 backbone 参数 & 固定 BN/Dropout
    network.requires_grad_(False)
    network.eval()
    visual_prompt.to(device)
    network.to(device)

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        # === (1) 估计/更新映射矩阵 ===
        mapping_matrix = _build_mapping_matrix(
            mapping_mode=mapping_mode,
            visual_prompt=visual_prompt,
            network=network,
            train_loader=loaders['train'],
            device=device,
            pretrain_num_classes=pretrain_num_classes,
            target_num_classes=target_num_classes,
            lap_blm=lap_blm,
            lap_blmp=lap_blmp,
            topk_ratio=topk_ratio
        )  # [P, C]

        # === (2) 训练 prompt ===
        visual_prompt.train()
        total_num, true_num, loss_sum = 0, 0, 0.0
        pbar = tqdm(loaders['train'], total=len(loaders['train']),
                    desc=f"Training Epo {epoch}", ncols=100)

        for step, (x, y) in enumerate(pbar):
            if device.type == "cuda":
                x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            else:
                x = x.to(device); y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            with get_amp_ctx(device):
                # backbone 前向不能 no_grad，否则梯度无法对输入求导从而更新 prompt
                logits_p = network(visual_prompt(x))                   # [B, P]
                probs_p = torch.softmax(logits_p / temp, dim=1)        # [B, P]
                fx = probs_p @ mapping_matrix                          # [B, C]
                loss = F.cross_entropy(fx, y, reduction='mean')

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()


            # 统计
            bs = y.size(0)
            total_num += bs
            loss_sum += loss.item() * bs
            true_num += (fx.argmax(1) == y).float().sum().item()

            if log_interval and (step + 1) % log_interval == 0:
                avg_loss = loss_sum / total_num
                pbar.set_postfix_str(f"Train Acc {100*true_num/total_num:.2f}% | Loss {avg_loss:.4f}")
            else:
                pbar.set_postfix_str(f"Train Acc {100*true_num/total_num:.2f}%")

        scheduler.step()

        # === (3) 测试评估 ===
        visual_prompt.eval()
        total_num, true_num = 0, 0
        pbar = tqdm(loaders['test'], total=len(loaders['test']),
                    desc=f"Testing Epo {epoch}", ncols=100)

        with torch.no_grad():
            for x, y in pbar:
                if device.type == "cuda":
                    x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                else:
                    x = x.to(device); y = y.to(device)
                # x = x.to(device, non_blocking=True)
                # y = y.to(device, non_blocking=True)
                logits_p = network(visual_prompt(x))                   # [B, P]
                probs_p = torch.softmax(logits_p / temp, dim=1)        # [B, P]
                fx = probs_p @ mapping_matrix                          # [B, C]
                total_num += y.size(0)
                true_num += (fx.argmax(1) == y).float().sum().item()
                acc = true_num / total_num
                pbar.set_postfix_str(f"Testing Acc {100*acc:.2f}%, Best Acc {100*best_acc:.2f}%")

        # === (4) 记录最优并可选保存 ===
        epoch_acc = true_num / max(1, total_num)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_state = {
                "visual_prompt": {k: v.detach().cpu() for k, v in visual_prompt.state_dict().items()},
                "mapping_matrix": mapping_matrix.detach().cpu(),
                "epoch": epoch,
                "acc": best_acc,
                "temp": temp,
                "mapping_mode": mapping_mode,
            }
            if save_best_path:
                os.makedirs(os.path.dirname(save_best_path), exist_ok=True)
                torch.save(best_state, save_best_path)

    return best_acc, best_state



# ===== Trainable Linear Layer Classifier =====
class TrainableLinearClassifier(nn.Module):
    def __init__(self, blackbox_model,visual_prompt,input_dim=1000, output_dim=10):
        super().__init__()
        self.blackbox = blackbox_model
        self.linear = nn.Linear(input_dim, output_dim)
        self.visual_prompt = visual_prompt
    def forward(self, x):
        x = self.visual_prompt(x)
        fx = self.blackbox(x)
        return self.linear(fx)




# ===== Evaluation =====
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            if device.type == "cuda":
                x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            else:
                x = x.to(device); y = y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total



# ===== 训练并评估线性探测层 =====
def train_linear_probe(
    model,                          # 包含 linear（必须）与可选 visual_prompt
    loaders,                        # {'train': DataLoader, 'test': DataLoader}
    epochs=5,                       # 轮数
    optimizer=None,                 # 若为 None，则自动从 model.linear / model.visual_prompt 收集参数创建 Adam
    scheduler=None,                 # 可选 lr scheduler（按 epoch 调用 step）
    device="cuda",
    save_best_path=None,            # 若提供路径，则保存最优（linear + 可选 VP）权重
    log_interval=0,                 # >0 时每隔若干 step 打印一次 loss
    lr=1e-3                         # 当 optimizer 为 None 时使用
):
    """
    返回: best_acc(float), best_state(dict or None)
    打印风格：
      - 训练：desc=f"Training Epo {epoch}"，postfix 显示 "Train Acc xx.xx% | Loss yyyy"
      - 测试：desc=f"Testing Epo {epoch}"，postfix 显示 "Testing Acc xx.xx%, Best Acc yy.yy%"
    """
    import os
    import torch
    import torch.nn.functional as F
    from torch import nn
    from tqdm import tqdm

    # --- device & to(device) ---
    model.to(device)

    # --- 组装可训练参数（兼容无 VP） ---
    params = []
    if hasattr(model, "linear") and isinstance(model.linear, nn.Module):
        params += [p for p in model.linear.parameters() if p.requires_grad]
    if hasattr(model, "visual_prompt") and isinstance(model.visual_prompt, nn.Module):
        params += [p for p in model.visual_prompt.parameters() if p.requires_grad]

    if optimizer is None:
        optimizer = torch.optim.Adam(params, lr=lr) if len(params) > 0 else None

    # --- AMP 兼容（与 mapping 的 GradScaler 对齐） ---
    try:
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler() if (hasattr(device, "type") and device.type == "cuda" and torch.cuda.is_available()) else None
        def _amp_ctx():
            return autocast(enabled=(hasattr(device, "type") and device.type == "cuda" and torch.cuda.is_available()))

    except Exception:
        import contextlib
        scaler = None
        def _amp_ctx():
            return contextlib.nullcontext()

    best_acc = 0.0
    best_state = None
    loss_fn = nn.CrossEntropyLoss()

    assert 'train' in loaders and 'test' in loaders, "loaders 需要包含 'train' 与 'test'"

    for epoch in range(epochs):
        # === (1) 训练 ===
        model.train()
        total_num, true_num, loss_sum = 0, 0, 0.0
        pbar = tqdm(loaders['train'], total=len(loaders['train']),
                    desc=f"Training Epo {epoch}", ncols=100)

        for step, (x, y) in enumerate(pbar):
            if device.type == "cuda":
                x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            else:
                x = x.to(device); y = y.to(device)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            with _amp_ctx():
                logits = model(x)                 # 线性探测前向（若含 VP，模型内部会用到）
                loss = loss_fn(logits, y)

            # 反传（若存在可训练参数）
            if optimizer is not None:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            # 统计
            bs = y.size(0)
            total_num += bs
            loss_sum += loss.item() * bs
            true_num += (logits.argmax(1) == y).float().sum().item()

            if log_interval and (step + 1) % log_interval == 0:
                avg_loss = loss_sum / total_num
                pbar.set_postfix_str(f"Train Acc {100*true_num/total_num:.2f}% | Loss {avg_loss:.4f}")
            else:
                pbar.set_postfix_str(f"Train Acc {100*true_num/total_num:.2f}%")

        if scheduler is not None:
            scheduler.step()

        # === (2) 测试评估 ===
        model.eval()
        total_num, true_num = 0, 0
        pbar = tqdm(loaders['test'], total=len(loaders['test']),
                    desc=f"Testing Epo {epoch}", ncols=100)
        with torch.no_grad():
            for x, y in pbar:
                if device.type == "cuda":
                    x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                else:
                    x = x.to(device); y = y.to(device)

                logits = model(x)
                total_num += y.size(0)
                true_num += (logits.argmax(1) == y).float().sum().item()
                acc = true_num / total_num
                pbar.set_postfix_str(f"Testing Acc {100*acc:.2f}%, Best Acc {100*best_acc:.2f}%")

        # === (3) 记录最优并可选保存 ===
        epoch_acc = true_num / max(1, total_num)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            state = {"epoch": epoch, "acc": best_acc}

            # 只保存线性头与（若有）VP，避免把整个大 backbone 存进去
            if hasattr(model, "linear") and isinstance(model.linear, nn.Module):
                state["linear"] = {k: v.detach().cpu() for k, v in model.linear.state_dict().items()}
            if hasattr(model, "visual_prompt") and isinstance(model.visual_prompt, nn.Module):
                state["visual_prompt"] = {k: v.detach().cpu() for k, v in model.visual_prompt.state_dict().items()}

            best_state = state
            if save_best_path:
                os.makedirs(os.path.dirname(save_best_path), exist_ok=True)
                torch.save(best_state, save_best_path)

    return best_acc, best_state


# —— 2) 最终评测（确定性评估） ——
def evaluate_linear(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            if device.type == "cuda":
                x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            else:
                x = x.to(device); y = y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total





# ===== Run Experiments =====
def run_experiments(args):
    # —— 设备兜底：优先 CUDA→MPS→CPU
    base = DEFAULT_DEVICE
    req = getattr(args, "device", None)
    try:
        ad = torch.device(req) if req else base
    except Exception:
        ad = base
    # 若用户传了与实际不可用设备不一致，回退到 base
    if ad.type != base.type:
        ad = base
    args.device = ad
    print(f"[DEV] args.device = {args.device}")
    # ==== 数据加载与预处理 ====
    train_loader, test_loader, class_names = process_input(args)

    # ==== 获取黑盒模型 ====
    blackbox_model = get_blackbox_model(args)

    # ==== 视觉提示 ====
    vp_mode = args.vp_mode
    visual_prompt = build_visual_prompt(
        mode=vp_mode,                    # 现在 config 里已改为支持 none/padding
        image_size=args.image_size,
        layers=5,
        patch_size=8,
        channels=3
    ).to(args.device)

    # 使用 Mapping / Linear Probe
    if args.mapping_mode.lower() == 'none':
        print("[Mapping] Disabled")
        input_dim = args.input_dim if hasattr(args, 'input_dim') else 1000
        output_dim = args.output_dim if hasattr(args, 'output_dim') else 10
        model_lp = TrainableLinearClassifier(
            blackbox_model=blackbox_model,
            visual_prompt=visual_prompt,
            input_dim=input_dim,
            output_dim=output_dim
        ).to(args.device)
        # 直接使用线性探测
        visual_prompt = DummyVisualPrompt()
        # 调用线性训练器
        loaders = {'train': train_loader, 'test': test_loader}
        best_acc, best_state = train_linear_probe(model_lp,loaders, epochs=50, lr=1e-3)
        print("[Linear Training] Completed")
        
        # —— 1) 恢复最优权重到当前模型 ——
        if best_state is not None:
            if 'linear' in best_state:
                model_lp.linear.load_state_dict(best_state['linear'])
            if 'visual_prompt' in best_state:      # 线性探测通常没有，这里兼容
                model_lp.visual_prompt.load_state_dict(best_state['visual_prompt'])
        else:
            print("No best_state returned, using last-epoch weights.")
        final_acc = evaluate_linear(model_lp, test_loader, args.device)
        print(f"[Linear Probe] Final Eval Acc: {final_acc:.2%} (should ≈ best_acc)")
    # 如果 mp存在 然后 vp 为none    
    elif args.vp_mode == 'none' and args.mapping_mode.lower() != 'none':
        print("[Mapping] Enabled, but VP is None. Using fixed mapping.")
        # 直接评估黑盒模型
        acc, mapping = eval_zero_shot_mapping(
        blackbox_model,
        loaders={'train': train_loader, 'test': test_loader},
        class_names=class_names,
        mapping_mode=args.mapping_mode,
        device=args.device,
        temp=0.6, lap_blm=1, lap_blmp=2, topk_ratio=0.02
        )
        print(f"[Zero-Shot Mapping] Test Acc = {acc*100:.2f}%")
    else:
        # 启用 model reprogramming
        print("[Mapping] Enabled,model reprogramming starting")
        print(f"[Visual Prompt] Mode: {vp_mode}")
        print(f"[Mapping] Enabled: {args.mapping_mode}")

        # ==== 规范化 mapping_mode ====
        mapping_mode = args.mapping_mode.lower()
        if mapping_mode == 'blm+':
            mapping_mode = 'blmp'            # 内部实现用 'blmp' 表示 BLM+

        # ==== Optimizer & Scheduler（只训练 VP）====
        # 这里要考虑vp可能为 none 或 DummyVisualPrompt

        optimizer = torch.optim.Adam(visual_prompt.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8], gamma=0.1)
        # ==== 训练：每个 epoch 动态重估映射矩阵 ====
        best_acc, best_state = train_visual_prompt_with_mapping(
            network=blackbox_model,
            visual_prompt=visual_prompt,
            loaders={'train': train_loader, 'test': test_loader},
            epochs=50,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
            mapping_mode=mapping_mode,
            class_names=class_names,
            pretrain_num_classes=args.input_dim,
            temp=0.6,
        )

        # ==== 用最优权重+映射做评估 ====
        if best_state is not None:
            visual_prompt.load_state_dict(best_state["visual_prompt"])
            fixed_model = FixedMappingClassifier(
                blackbox_model,
                best_state["mapping_matrix"].to(args.device),
                visual_prompt,
                temp=best_state["temp"],
            ).to(args.device)
            acc_fixed = evaluate(fixed_model, test_loader, args.device)
            print(f"[Fixed Mapping + VP] Best Test Acc = {acc_fixed*100:.2f}%")
        else:
            print("No best_state saved. Check training loop.")



 
print("---- watermark + BLM ----")
from config import Config

# print(args)

args = Config([
"--vp_mode", "watermark",
"--model_name", "resnet18",
"--mapping_mode", "blm",
"--dataset", "cifar10",
"--input_dim","1000",
"--output_dim","10",
"--train_percent", "80",   
"--seed", "42"
]).get()
run_experiments(args)
# print(args)

args = Config([
"--vp_mode", "watermark",
"--model_name", "resnet18",
"--mapping_mode", "blm",
"--dataset", "cifar10",
"--input_dim","1000",
"--output_dim","10",
"--train_percent", "60",   
"--seed", "42"
]).get()
run_experiments(args)
# print(args)

args = Config([
"--vp_mode", "watermark",
"--model_name", "resnet18",
"--mapping_mode", "blm",
"--dataset", "cifar10",
"--input_dim","1000",
"--output_dim","10",
"--train_percent", "40",   
"--seed", "42"
]).get()
run_experiments(args)
# print(args)

args = Config([
"--vp_mode", "watermark",
"--model_name", "resnet18",
"--mapping_mode", "blm",
"--dataset", "cifar10",
"--input_dim","1000",
"--output_dim","10",
"--train_percent", "201",   
"--seed", "42"
]).get()
run_experiments(args)
# print(args)


print("---- watermark + BLM (cifar100) ----")
from config import Config
args = Config([
"--vp_mode", "watermark",
"--model_name", "resnet18",
"--mapping_mode", "blm",
"--dataset", "cifar100",
"--input_dim","1000",
"--output_dim","100",
"--train_percent", "100",   
"--seed", "42"
]).get()
run_experiments(args)
# print(args)


args = Config([
"--vp_mode", "watermark",
"--model_name", "resnet18",
"--mapping_mode", "blm",
"--dataset", "cifar100",
"--input_dim","1000",
"--output_dim","100",
"--train_percent", "80",   
"--seed", "42"
]).get()
run_experiments(args)
# print(args)

args = Config([
"--vp_mode", "watermark",
"--model_name", "resnet18",
"--mapping_mode", "blm",
"--dataset", "cifar100",
"--input_dim","1000",
"--output_dim","100",
"--train_percent", "60",   
"--seed", "42"
]).get()
run_experiments(args)
# print(args)

args = Config([
"--vp_mode", "watermark",
"--model_name", "resnet18",
"--mapping_mode", "blm",
"--dataset", "cifar100",
"--input_dim","1000",
"--output_dim","100",
"--train_percent", "40",   
"--seed", "42"
]).get()
run_experiments(args)
# print(args)

args = Config([
"--vp_mode", "watermark",
"--model_name", "resnet18",
"--mapping_mode", "blm",
"--dataset", "cifar100",
"--input_dim","1000",
"--output_dim","100",
"--train_percent", "20",   
"--seed", "42"
]).get()
run_experiments(args)
# print(args)




















