# training.py
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from torch.cuda.amp import GradScaler

# 从项目其他模块导入所需函数和类
from mapping import one2one_mappnig_matrix, blm_reweight_matrix, blmp_reweight_matrix
from utils import tqdm, get_amp_ctx
from models import FixedMappingClassifier

# ===================================================================
# ===== 1. 通用评估函数
# ===================================================================

def evaluate(model, loader, device):
    """
    在给定数据集上评估模型的准确率。

    Args:
        model (nn.Module): 待评估的模型。
        loader (DataLoader): 数据加载器。
        device (torch.device): 计算设备 (CPU or CUDA)。

    Returns:
        float: 模型在数据集上的准确率。
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / max(1, total)


# ===================================================================
# ===== 2. 标签映射 (Label Mapping) 相关函数
# ===================================================================

@torch.no_grad()
def _build_mapping_matrix(mapping_mode, visual_prompt, network, train_loader, device,
                          pretrain_num_classes, target_num_classes, topk_ratio=0.02,
                          lap_blm=1, lap_blmp=2):
    """
    根据指定的映射模式，计算从源域到目标域的 [P, C] 映射矩阵。

    Args:
        mapping_mode (str): 映射模式 ('ilm', 'blm', 'blm+').
        visual_prompt (nn.Module): 当前的视觉提示模块。
        network (nn.Module): 冻结的黑盒模型。
        train_loader (DataLoader): 用于估计映射的训练数据加载器。
        device (torch.device): 计算设备。
        pretrain_num_classes (int): 预训练模型的类别数 (P)。
        target_num_classes (int): 目标任务的类别数 (C)。
        topk_ratio (float): BLMP+ 中 top-k 的比例。
        lap_blm (int): BLM 的拉普拉斯平滑因子。
        lap_blmp (int): BLMP+ 的拉普拉斯平滑因子。

    Returns:
        torch.Tensor: 计算好的映射矩阵。
    """
    # 规范化映射名称
    mode = mapping_mode.lower().replace('blm+', 'blmp')

    if mode == 'ilm':
        # ILM 返回索引序列，需转换为 one-hot 矩阵
        seq = one2one_mappnig_matrix(visual_prompt, network, train_loader)
        mapping = torch.zeros(pretrain_num_classes, target_num_classes, device=device)
        mapping[seq, torch.arange(target_num_classes, device=device)] = 1.0
        return mapping.float()
    
    elif mode == 'blm':
        return blm_reweight_matrix(visual_prompt, network, train_loader, lap=lap_blm).to(device).float()
    
    elif mode == 'blmp':
        k = max(1, int(pretrain_num_classes * topk_ratio))
        return blmp_reweight_matrix(visual_prompt, network, train_loader, lap=lap_blmp, k=k).to(device).float()
    
    else:
        raise ValueError(f"未知的映射模式: {mapping_mode}")


def eval_zero_shot_mapping(blackbox_model, loaders, class_names, mapping_mode, device, temp=0.6):
    """
    执行 Zero-shot 评估：不训练任何参数，仅计算一次映射矩阵并测试其性能。
    """
    # Zero-shot 场景下不使用视觉提示
    from vp import DummyVisualPrompt
    
    visual_prompt = DummyVisualPrompt().to(device)
    P = getattr(getattr(blackbox_model, "fc", None), "out_features", 1000)
    C = len(class_names)

    print("构建 Zero-shot 映射矩阵...")
    mapping = _build_mapping_matrix(
        mapping_mode, visual_prompt, blackbox_model, loaders['train'], device, P, C
    )
    
    classifier = FixedMappingClassifier(blackbox_model, mapping, visual_prompt, temp).to(device)
    
    print("在测试集上进行评估...")
    acc = evaluate(classifier, loaders.get('test', loaders.get('val')), device)
    return acc, mapping


# ===================================================================
# ===== 3. 核心训练循环
# ===================================================================

def train_visual_prompt_with_mapping(network, visual_prompt, loaders, epochs, optimizer, scheduler,
                                     device, mapping_mode, class_names, pretrain_num_classes,
                                     temp=0.6, save_best_path=None, writer=None):
    """
    模型重编程训练循环：训练 Visual Prompt，并在每个 epoch 动态更新 Mapping 矩阵。
    """
    target_num_classes = len(class_names)
    scaler = GradScaler()
    best_acc, best_state = 0.0, None

    # 冻结 backbone
    network.requires_grad_(False)
    network.eval()

    for epoch in range(epochs):
        # (1) 基于当前 VP 估计/更新映射矩阵
        mapping_matrix = _build_mapping_matrix(
            mapping_mode, visual_prompt, network, loaders['train'], device,
            pretrain_num_classes, target_num_classes
        )
        
        # (2) 训练 prompt
        visual_prompt.train()
        total_loss = 0
        pbar = tqdm(loaders['train'], desc=f"VP Training Epo {epoch+1}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with get_amp_ctx(device):
                # backbone 前向不能 no_grad，否则梯度无法对输入求导
                logits_p = network(visual_prompt(x))
                probs_p = torch.softmax(logits_p / temp, dim=1)
                fx = probs_p @ mapping_matrix
                loss = F.cross_entropy(fx, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        # (3) 使用当前 epoch 的映射矩阵进行测试评估
        classifier = FixedMappingClassifier(network, mapping_matrix, visual_prompt, temp).to(device)
        epoch_acc = evaluate(classifier, loaders['test'], device)
        
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss / len(pbar):.4f} | Test Acc: {epoch_acc*100:.2f}% | Best Acc: {best_acc*100:.2f}%")

        # (4) TensorBoard 日志记录
        if writer:
            writer.add_scalar('Loss/train_epoch', total_loss / len(pbar), epoch + 1)
            writer.add_scalar('Accuracy/test', epoch_acc, epoch + 1)
            writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch + 1)

        # (5) 记录并保存最优模型状态
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_state = {
                "visual_prompt": {k: v.detach().cpu() for k, v in visual_prompt.state_dict().items()},
                "mapping_matrix": mapping_matrix.detach().cpu(),
                "epoch": epoch + 1,
                "acc": best_acc,
                "temp": temp,
                "mapping_mode": mapping_mode,
            }
            if save_best_path:
                os.makedirs(os.path.dirname(save_best_path) or '.', exist_ok=True)
                torch.save(best_state, save_best_path)
                print(f"新最佳模型已保存至: {save_best_path}")
        
        # Scheduler Step要在最后，避免Pytorch 1.x版本的警告
        scheduler.step()

    return best_acc, best_state


def train_linear_probe(model, loaders, epochs, lr, device, save_best_path=None, writer=None):
    """
    线性探测训练循环：只训练分类器头部 (以及可选的 VP)。
    """
    # 自动识别需要训练的参数
    params_to_train = []
    if hasattr(model, 'linear') and isinstance(model.linear, nn.Module):
        params_to_train.extend(model.linear.parameters())
    if hasattr(model, 'visual_prompt'):
        # 假设 DummyVisualPrompt 没有参数或 requires_grad=False
        params_to_train.extend(p for p in model.visual_prompt.parameters() if p.requires_grad)

    if not params_to_train:
        print("警告: 线性探测模式下没有找到可训练的参数。")
        return evaluate(model, loaders['test'], device), None

    optimizer = torch.optim.Adam(params_to_train, lr=lr)
    scaler = GradScaler()
    loss_fn = torch.nn.CrossEntropyLoss()
    best_acc, best_state = 0.0, None

    for epoch in range(epochs):
        model.train()
        if hasattr(model, 'blackbox'): model.blackbox.eval() # 确保黑盒模型处于评估模式

        total_loss = 0
        pbar = tqdm(loaders['train'], desc=f"LP Training Epo {epoch+1}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with get_amp_ctx(device):
                logits = model(x)
                loss = loss_fn(logits, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        epoch_acc = evaluate(model, loaders['test'], device)
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss / len(pbar):.4f} | Test Acc: {epoch_acc*100:.2f}% | Best Acc: {best_acc*100:.2f}%")

        # TensorBoard 日志记录
        if writer:
            writer.add_scalar('Loss/train_epoch', total_loss / len(pbar), epoch + 1)
            writer.add_scalar('Accuracy/test', epoch_acc, epoch + 1)
            writer.add_scalar('Learning_Rate', lr, epoch + 1) # 记录固定的学习率

        # 记录并保存最优模型状态
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            state = {"epoch": epoch + 1, "acc": best_acc}
            # 只保存可训练部分的权重
            if hasattr(model, "linear"):
                state["linear"] = {k: v.detach().cpu() for k, v in model.linear.state_dict().items()}
            if hasattr(model, "visual_prompt") and any(p.requires_grad for p in model.visual_prompt.parameters()):
                state["visual_prompt"] = {k: v.detach().cpu() for k, v in model.visual_prompt.state_dict().items()}
            
            best_state = state
            if save_best_path:
                os.makedirs(os.path.dirname(save_best_path) or '.', exist_ok=True)
                torch.save(best_state, save_best_path)
                print(f"新最佳模型已保存至: {save_best_path}")
            
    return best_acc, best_state