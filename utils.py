# utils.py
import torch
import os
import sys
from contextlib import nullcontext
from tqdm import tqdm as _tqdm
from data import get_train_preprocess, prepare_additive_data
import random
import numpy as np

# --- TQDM 配置 ---
def tqdm(*args, **kwargs):
    """
    在非交互式环境 (如 Slurm) 中自动禁用 tqdm 进度条
    """
    is_non_interactive = (
        not sys.stdout.isatty()
        or "SLURM_JOB_ID" in os.environ
        or "PBS_JOBID" in os.environ
        or "LSB_JOBID" in os.environ
        or os.environ.get("TQDM_DISABLE") == "1"
    )
    kwargs.setdefault("disable", is_non_interactive)
    kwargs.setdefault("dynamic_ncols", True)
    return _tqdm(*args, **kwargs)

# --- AMP 上下文 ---
def get_amp_ctx(device):
    """获取 AMP (自动混合精度) 的上下文管理器"""
    use_cuda = torch.cuda.is_available() and ("cuda" in str(device))
    if use_cuda:
        return torch.amp.autocast("cuda")
    else:
        return nullcontext()

# --- 数据处理 ---
def process_input(args):
    """根据配置加载和准备数据集"""
    train_transform, test_transform = get_train_preprocess(args.vp_mode)
    loaders, class_names = prepare_additive_data(
        dataset=args.dataset,
        data_path=args.data_root,
        preprocess=train_transform,
        test_process=test_transform,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        train_percent=args.train_percent,
        seed=getattr(args, "seed", 42)
    )
    return loaders['train'], loaders['test'], class_names

# --- 可复现性 ---
def set_seed(seed):
    """设置全局随机种子以保证实验可复现"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 可选: 确保 cuDNN 的确定性行为，可能会影响性能
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False