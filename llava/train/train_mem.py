import os
os.environ["WANDB_API_KEY"] = "7e72b8584c884832f443d2dab6e23c9f8d282ca1"
os.environ["WANDB_MODE"] = "offline"

import sys
sys.setrecursionlimit(10000)
sys.path.append('.')
sys.path.append('..')

# Need to call this before importing transformers.
from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()

from llava.train.train import train

if __name__ == "__main__":
    train()
