import os
import sys
from datetime import datetime

# 假设您的主代码和Config类在名为 a.py 的文件中
# 如果您的文件名不同，请相应修改下面的 'a'
try:
    from main import Config, run_experiments
except ImportError as e:
    print(f"错误: 无法从您的主代码文件中导入 'Config' 或 'run_experiments'。")
    print(f"ImportError: {e}")
    print("请确保此脚本与您的主代码文件（包含Config和run_experiments函数）在同一目录下。")
    sys.exit(1)

# ==============================================================================
# 1. 实验配置
# ==============================================================================

# 定义实验组
# 格式为: (源域, 目标域)
EXP_GROUPS = {
    "小距离 (Short Distance)": [
        ("real", "painting"),
        ("painting", "real"),
        ("clipart", "sketch"),
        ("sketch", "clipart"),
    ],
    "中距离 (Medium Distance)": [
        ("real", "sketch"), ("sketch", "real"),
        ("real", "infograph"), ("infograph", "real"),
        ("clipart", "painting"), ("painting", "clipart"),
        ("painting", "sketch"), ("sketch", "painting"),
        ("infograph", "painting"), ("painting", "infograph"),
    ],
    "大距离 (Long Distance)": [
        ("real", "quickdraw"), ("quickdraw", "real"),
        ("sketch", "quickdraw"), ("quickdraw", "sketch"),
        ("clipart", "quickdraw"), ("quickdraw", "clipart"),
        ("painting", "quickdraw"), ("quickdraw", "painting"),
        ("infograph", "quickdraw"), ("quickdraw", "infograph"),
    ],
}

# 定义要测试的方法
# 键为方法简称，值为对应的参数设置
METHODS = {
    # 纯线性探测
    "LP": {
        "mapping_mode": "none",
        "vp_mode": "none"
    },
    # 带视觉提示的线性探测
    "SSM+LP": {
        "mapping_mode": "none",
        "vp_mode": "instance"
    },
    # 带视觉提示的标签映射
    "SSM+BLM+": {
        "mapping_mode": "blm+",
        "vp_mode": "instance"
    },
}

# 定义基础参数 (请根据您的环境修改)
# ★★★★★ 请务必确认您的数据根目录是否正确 ★★★★★
DATA_ROOT = "/data/gpfs/projects/punim1943"
BASE_ARGS_TEMPLATE = [
    "--model_name", "resnet18",
    "--data_root", DATA_ROOT,
    "--output_dim", "345",
    "--train_percent", "100",
    "--seed", "42",
]


# ==============================================================================
# 2. 核心执行逻辑
# ==============================================================================

def main():
    """主函数，按顺序执行所有实验"""
    start_time = datetime.now()
    print(f"实验开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    total_exp_count = sum(len(pairs) for pairs in EXP_GROUPS.values()) * len(METHODS)
    current_exp = 0

    # 遍历所有实验组
    for group_name, pairs in EXP_GROUPS.items():
        # 遍历组内的所有配对
        for source_domain, target_domain in pairs:
            # 遍历所有要测试的方法
            for method_name, method_params in METHODS.items():
                current_exp += 1
                header = f"🧪 实验 {current_exp}/{total_exp_count}: [{group_name}] - [{method_name}] | {source_domain} → {target_domain} 🧪"
                print(f"\n{'='*len(header)}\n{header}\n{'='*len(header)}")

                # 1. 复制基础参数模板
                args_list = BASE_ARGS_TEMPLATE[:]

                # 2. 添加与当前配对相关的参数
                args_list.extend(["--dataset", f"domainnet/{target_domain}"])
                args_list.extend(["--weight_path", f"./model/domainnet_{source_domain}_resnet18.pth"])

                # 3. 添加与当前方法相关的参数
                for key, value in method_params.items():
                    args_list.extend([f"--{key}", str(value)])

                # 4. 动态设置核心的 input_dim 参数
                if method_params["mapping_mode"] == "none": # LP 和 SSM+LP
                    args_list.extend(["--input_dim", "512"]) # 骨干网络特征维度
                else: # SSM+BLM+
                    args_list.extend(["--input_dim", "345"])  # 上游任务类别数

                try:
                    # 5. 创建Config并运行实验
                    print("当前实验配置:", " ".join(args_list))
                    args = Config(args_list).get()
                    run_experiments(args)
                except FileNotFoundError as e:
                    print(f"❌ 实验跳过: 权重文件不存在！请先完成所有上游模型的训练。")
                    print(f"   缺失文件: {e.filename}")
                except Exception as e:
                    print(f"❌ 实验出错: {e}")
                    print("   跳过此实验，继续下一个...")
    
    end_time = datetime.now()
    print(f"\n{'='*80}")
    print(f"🎉 所有实验已完成！")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {end_time - start_time}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()