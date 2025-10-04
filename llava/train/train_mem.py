import os
import sys
import subprocess
# os.environ["WANDB_API_KEY"] = "xxx"
# os.environ["WANDB_MODE"] = "offline"
sys.setrecursionlimit(10000)
sys.path.append('.')
sys.path.append('..')
import warnings
warnings.filterwarnings('ignore')

def export_env_info():
    import platform
    print("Python version:", platform.python_version())
    print("\nInstalled packages:")
    subprocess.run([sys.executable, "-m", "pip", "freeze"])
    try:
        import torch
        print("\nCUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA version:", torch.version.cuda)
            print("cuDNN version:", torch.backends.cudnn.version())
            print("GPU count:", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("\nPyTorch not installed, skipping CUDA info.")

if __name__ == "__main__":
    export_env_info()
    from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    replace_llama_attn_with_flash_attn()
    from llava.train.train import train
    train()