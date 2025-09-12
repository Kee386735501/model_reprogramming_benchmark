# main.py
import torch
import os
import time
from torch.utils.tensorboard import SummaryWriter  # <-- 1. 导入 SummaryWriter

from config import Config
from utils import process_input, set_seed
from models import get_blackbox_model, TrainableLinearClassifier, FixedMappingClassifier
from vp import build_visual_prompt, DummyVisualPrompt
from training import train_linear_probe, train_visual_prompt_with_mapping, eval_zero_shot_mapping, evaluate

def run_experiments(args):
    """
    协调并运行指定配置的实验
    """
    # ==================== TensorBoard 初始化 ====================
    # 2. 为每次运行创建一个唯一的日志目录
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # 构建一个清晰的文件夹名称，例如: cifar100_blm_instance_20250910-183000
    log_dir_name = f"{args.dataset}_{args.mapping_mode}_{args.vp_mode}_{timestamp}"
    log_dir = os.path.join("logs", "tensorboard", log_dir_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志将保存至: {log_dir}")
    # ==========================================================

    # 1. 数据加载与预处理
    train_loader, test_loader, class_names = process_input(args)
    loaders = {'train': train_loader, 'test': test_loader}

    # 2. 获取黑盒模型
    blackbox_model = get_blackbox_model(args)

    # 3. 构建视觉提示
    visual_prompt = build_visual_prompt(
        mode=args.vp_mode, image_size=args.image_size
    ).to(args.device)

    best_acc = 0.0 # 初始化 best_acc
    
    # 4. 根据 mapping_mode 选择实验分支
    # ==================== 分支 A: 线性探测 ====================
    if args.mapping_mode.lower() == 'none':
        print("[Mode] Running Linear Probe.")
        model_lp = TrainableLinearClassifier(
            blackbox_model=blackbox_model, visual_prompt=visual_prompt,
            input_dim=args.input_dim, output_dim=args.output_dim
        ).to(args.device)
        
        # 3. 将 writer 传入训练函数
        best_acc, _ = train_linear_probe(model_lp, loaders, epochs=50, lr=1e-3, device=args.device, writer=writer)
        print(f"[Result] Linear Probe Best Test Accuracy: {best_acc*100:.2f}%")

    # ==================== 分支 B: Zero-Shot Mapping ====================
    elif args.vp_mode == 'none' and args.mapping_mode.lower() != 'none':
        print("[Mode] Running Zero-Shot Mapping (No VP).")
        best_acc, _ = eval_zero_shot_mapping(blackbox_model, loaders, class_names, args.mapping_mode, args.device)
        print(f"[Result] Zero-Shot Mapping Test Accuracy: {best_acc*100:.2f}%")

    # ==================== 分支 C: Model Reprogramming ====================
    else:
        print("[Mode] Running Model Reprogramming (VP + Mapping).")
        optimizer = torch.optim.Adam(visual_prompt.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8], gamma=0.1)
        
        # 3. 将 writer 传入训练函数
        # best_acc, _ = train_visual_prompt_with_mapping(
        #     network=blackbox_model, visual_prompt=visual_prompt, loaders=loaders,
        #     epochs=50, optimizer=optimizer, scheduler=scheduler, device=args.device,
        #     mapping_mode=args.mapping_mode, class_names=class_names,
        #     pretrain_num_classes=args.input_dim, writer=writer
        # )
        best_acc, _ = train_visual_prompt_with_mapping(
            network=blackbox_model, visual_prompt=visual_prompt, loaders=loaders,
            epochs=args.epochs, optimizer=optimizer, scheduler=scheduler, device=args.device,
            mapping_mode=args.mapping_mode, class_names=class_names,
            pretrain_num_classes=args.input_dim, writer=writer
        )
        print(f"[Result] Model Reprogramming Best Test Accuracy: {best_acc*100:.2f}%")

    # ==================== 记录超参数和最终结果 ====================
    # 4. 将所有参数转换为字符串以便记录
    hparams = {k: str(v) for k, v in vars(args).items()}
    writer.add_hparams(hparams, {'hparam/best_accuracy': best_acc})
    writer.close()
    print("TensorBoard 日志记录完毕。")
    # ==========================================================

if __name__ == '__main__':
    args = Config().get()
    set_seed(args.seed)
    
    print("="*50)
    print(f"Running experiment with configuration: {vars(args)}")
    print("="*50)
    
    run_experiments(args)