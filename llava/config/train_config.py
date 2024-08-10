import transformers
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    pass

@dataclass
class DataArguments:
    pass

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_vision_tower: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # data arguments
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_stage: str = field(default=None, metadata={"help": "Pretrain or finetune."})
    train_file: str = field(default=None, metadata={"help": "1,2,3..."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    # training arguments
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    # lora params
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
