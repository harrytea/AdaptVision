import os
import copy
import json
from typing import Dict, Sequence
from dataclasses import dataclass

import torch
import transformers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.config.data_config import PROMPT_DATA_PRE, PROMPT_DATA_TUNE
from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_image_token, sliding_window, expand2square
from torch.utils.data import Dataset
from PIL import Image

from torch.distributed import get_rank

def rank0_print(*args):
    if get_rank() == 0:
        print(*args)

# 在human的第一个问题中加入<image>标记: xxx --> <im_start><image><im_end>\n xxx
def preprocess_multimodal(sources, data_args):
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()

                # sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources



def preprocess_v1(sources, tokenizer, has_image=False):
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    #  from:  human; value: xxx
    #  from:  gpt; value: xxx
    conversations = []  # Apply prompt templates
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]  # 如果第一个问题不是human提出的，则移除这个问题

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]  # 获取对话的角色
            conv.append_message(role, sentence["value"])
            
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    
    sep = conv.sep + conv.roles[1] + ": "  # Mask targets  ### ' ASSISTANT: '
    for conversation, target in zip(conversations, targets):

        total_len = target.numel()  # add this

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len:cur_len+instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"preprocess_v1")
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}.  (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )



def preprocess(sources, tokenizer, has_image=False):
    """Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path, tokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()
        # prompt data
        list_data_dict = []
        if data_args.data_stage == "pretrain":
            for path in PROMPT_DATA_PRE:
                prompt_path = os.path.join(data_path, path)
                list_data_dict += json.load(open(prompt_path, "r"))
            rank0_print("Pretrain stage: ", PROMPT_DATA_PRE)
        else:
            for path in PROMPT_DATA_TUNE:
                prompt_path = os.path.join(data_path, path)
                list_data_dict += json.load(open(prompt_path, "r"))
            rank0_print("Finetuning stage: ", PROMPT_DATA_TUNE)
        rank0_print("Formatting inputs...Skip in lazy mode")

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        input_conv = self.list_data_dict[i]
        if 'image' in input_conv:
            data_folder = input_conv['image_folder']
            image_file = input_conv['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, data_folder, image_file)).convert('RGB')
            # image = image.resize((1000, 2000))

            if self.data_args.image_aspect_ratio == 'square':
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            elif self.data_args.image_aspect_ratio == 'pad':
                process_size = processor.crop_size['height']
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean), process_size)
                # image = np.(image.resize((1200, 1300)))
                windows_img, windows_index = sliding_window(image, stride=process_size)
                image_concat = []
                for img in windows_img:
                    img = processor.preprocess(img, return_tensors='pt')['pixel_values'][0]
                    image_concat.append(img)
                image = torch.stack(image_concat, dim=0)
                # image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(copy.deepcopy([input_conv["conversations"]]), self.data_args)
        else:
            sources = copy.deepcopy([input_conv["conversations"]])

        data_dict = preprocess(sources, self.tokenizer, has_image=('image' in input_conv))
        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])


        for idx, win_index in enumerate(windows_index):
            joined_str = ",".join(str(num) for num in win_index)
            index_str = f"<{joined_str}>"
            tensor_index = self.tokenizer.encode(index_str)[1]
            windows_index[idx] = torch.tensor(tensor_index, dtype=torch.long)
        # image exist in the data
        data_dict['image_index'] = windows_index
        if 'image' in input_conv:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:  # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size 
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning.
        pad the sequence in the same length
    """
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id),)

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            # if all(x is not None and x.shape == images[0].shape for x in images):
                # batch['images'] = (torch.stack(images), [instance['image_index'] for instance in instances])
            # else:
                # batch['images'] = (images, [instance['image_index'] for instance in instances])
            batch['images'] = (images, [instance['image_index'] for instance in instances])
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
