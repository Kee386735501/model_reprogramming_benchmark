# training.py
from functools import partial
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from torch.cuda.amp import GradScaler,autocast
from const import config_vm
# 从项目其他模块导入所需函数和类
from mapping import label_mapping_base, label_mapping_calculation, one2one_mappnig_matrix, blm_reweight_matrix, blmp_reweight_matrix
from utils import tqdm, get_amp_ctx
from models import FixedMappingClassifier, get_blackbox_model

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

def train_visual_prompt_with_mapping(
            network,
            visual_prompt,
            loaders,
            epochs, 
            optimizer, 
            scheduler, 
            device,
            mapping_mode, 
            class_names,
            pretrain_num_classes, 
            writer
        ):
    """
    模型重编程训练循环：训练 Visual Prompt，并在每个 epoch 动态更新 Mapping 矩阵。
    """
    scaler = GradScaler()
    best_acc, best_state = 0.0, None
    # 冻结 network
    network.requires_grad_(False)
    network.eval()

    # Assuming config_vm is globally available
    # It's better practice to pass it as an argument, but this will work if it's imported.

    for epoch in range(epochs):
        # Label Mapping for ILM, BLM, BLM++
        if mapping_mode == 'ilm':
            mapping_matrix = one2one_mappnig_matrix(visual_prompt, network, loaders['train'])
            label_mapping = partial(label_mapping_base, mapping_sequence=mapping_matrix)
        elif mapping_mode == 'blm':
            mapping_matrix = blm_reweight_matrix(visual_prompt, network, loaders['train'], lap=config_vm['blm']['lap'])
            label_mapping = partial(label_mapping_calculation, mapping_matrix=mapping_matrix)
        elif mapping_mode == 'blmp':
            mapping_matrix = blmp_reweight_matrix(visual_prompt, network, loaders['train'], lap=config_vm['blmp']['lap'], k=int(len(class_names) * config_vm['blmp']['topk_ratio']))
            label_mapping = partial(label_mapping_calculation, mapping_matrix=mapping_matrix)
        
        # (2) 训练 prompt
        visual_prompt.train()
        train_loss_sum, train_total_num, train_true_num = 0, 0, 0
        pbar_train = tqdm(loaders['train'], total=len(loaders['train']), desc=f"Training Epo {epoch + 1}", ncols=100)
        
        for x, y in pbar_train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                fx = label_mapping(network(visual_prompt(x)))
                loss = F.cross_entropy(fx, y, reduction='mean')

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_total_num += y.size(0)
            train_true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            train_loss_sum += loss.item() * fx.size(0)
            
            train_acc = train_true_num / train_total_num
            pbar_train.set_postfix_str(f"Training Acc {100 * train_acc:.2f}%")

        avg_train_loss = train_loss_sum / train_total_num
        
        # (3) 测试
        visual_prompt.eval()
        test_true_num, test_total_num = 0, 0
        
        pbar_test = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Testing Epo {epoch + 1}", ncols=100)
        for x, y in pbar_test:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                fx = label_mapping(network(visual_prompt(x)))
            
            test_total_num += y.size(0)
            test_true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            
            # This is the correct way to calculate accuracy in the test loop
            test_acc = test_true_num / test_total_num
            pbar_test.set_postfix_str(f"Testing Acc {100 * test_acc:.2f}%, Best Acc {100 * best_acc:.2f}%")

        # (4) 打印并记录日志
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_train_loss:.4f} | Test Acc: {test_acc*100:.2f}% | Best Acc: {best_acc*100:.2f}%")
        if writer:
            writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch + 1)
            writer.add_scalar('Accuracy/test', test_acc, epoch + 1)
            writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch + 1)

        # (5) 更新最佳准确率
        if test_acc > best_acc:
            best_acc = test_acc
            
        scheduler.step()
    
    # Returning best_state which is None, which is fine if saving is not needed.
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