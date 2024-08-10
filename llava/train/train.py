import os
import torch
import transformers

from llava.train.llava_trainer import LLaVATrainer
from llava.config.train_config import ModelArguments, DataArguments, TrainingArguments

from llava import conversation as conversation_lib
from llava.model import *
from llava.utils import *
from llava.data import *

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    _, _, args = parser.parse_args_into_dataclasses()
    local_rank = args.local_rank  # 获取当前rank

    model = LlavaLlamaForCausalLM.from_pretrained(args.model_name_or_path)
    model.config.use_cache = False

    # gradient checkpointing
    if args.gradient_checkpointing:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


    # LoRA
    if args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if args.bits == 16:
            if args.bf16:
                model.to(torch.bfloat16)
            if args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)


    # tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )


    tokenizer.pad_token = tokenizer.unk_token


    model.get_model().initialize_vision_modules(model_args=args)
    vision_tower1 = model.get_model().get_vision_tower1()
    vision_tower1.to(dtype=torch.bfloat16, device=args.device)
    vision_tower2 = model.get_model().get_vision_tower2()
    vision_tower2.to(dtype=torch.bfloat16, device=args.device)

    model.get_model().update_config(args)

    model.get_model().initialize_adapter_modules(model_args=args)
    model.get_model().mm_projector.to(dtype=torch.bfloat16, device=args.device)
    model.get_model().down_vision.to(dtype=torch.bfloat16, device=args.device)
    args.image_processor = vision_tower1.image_processor
    args.is_multimodal = True
    model.to(dtype=torch.bfloat16, device=args.device)



    model.config.image_aspect_ratio = args.image_aspect_ratio
    model.config.tune_mm_mlp_adapter = args.tune_mm_mlp_adapter
    if args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        for p in model.get_model().down_vision.parameters():
            p.requires_grad = True
    # model.requires_grad_(False)
    # for p in model.get_model().mm_projector.parameters():
        # p.requires_grad = True

    if args.data_stage == "finetune":
        for p in model.parameters():
            p.requires_grad = True
        # for p in model.get_model().get_vision_tower1().parameters():
            # p.requires_grad = False
    # for n,p in model.named_parameters():
    #     if p.requires_grad == True:
    #         print(n)


    model.config.mm_use_im_start_end = args.mm_use_im_start_end
    model.initialize_vision_tokenizer(args, tokenizer=tokenizer)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=args)
    trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=args, **data_module)  # Trainer

    trainer.train()  # start training
    trainer.save_state()

    model.config.use_cache = True

    if args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if args.local_rank == 0 or args.local_rank == -1:
            model.config.save_pretrained(args.output_dir)
            model.save_pretrained(args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=args.output_dir)

if __name__ == "__main__":
    train()
