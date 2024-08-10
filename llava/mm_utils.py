from PIL import Image

import torch
from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX



def sliding_window(img, stride):
    width, height = img.size
    split_images = []
    split_index = []
    split_size = stride
    # 切割图像
    img_all = img.resize((int(stride), int(stride)), Image.LANCZOS)

    split_images.append(img_all)
    split_index.append((int(0), int(0)))

    if width <= stride and height <= stride:
        return split_images, split_index
    for i in range(0, height, split_size):
        for j in range(0, width, split_size):
            box = (j, i, j+split_size, i+split_size) # 定义裁剪区域
            split_image = img.crop(box)
            # split_image.save(f'result{i}_{j}.png')
            split_images.append(split_image)
            split_index.append((int((i+split_size)/stride), int((j+split_size)/stride)))
            # print(box, (i+split_size)/stride, (j+split_size)/stride)
    return split_images, split_index


def resize_image(pil_img, max_length, shard_size):
    width, height = pil_img.size
    # 判断最大边，并计算新尺寸
    if width > height:
        new_width = max_length
        new_height = int(max_length * height / width)
        if new_height <= shard_size:
            new_height = shard_size
        elif shard_size*2 >= new_height > shard_size:
            new_height = shard_size*2
        else:
            new_height = shard_size*3
    else:
        new_height = max_length
        new_width = int(max_length * width / height)
        if new_width <= shard_size:
            new_width = shard_size
        elif shard_size*2 >= new_width > shard_size:
            new_width = shard_size*2
        else:
            new_width = shard_size*3


    resized_img = pil_img.resize((new_width, new_height), Image.LANCZOS)  ######## max length
    return resized_img


def custom_ceil(value, shard_size):
    if 0 <= value <= shard_size:
        return shard_size
    elif shard_size < value <= shard_size*2:
        return shard_size*2
    else:
        return shard_size*3


def expand2square(pil_img, background_color, shard_size):
    width, height = pil_img.size

    max_edge = max(width, height)
    if max_edge <= shard_size:
        re_img = resize_image(pil_img, shard_size, shard_size)
    elif shard_size*2 >= max_edge > shard_size:
        re_img = resize_image(pil_img, shard_size*2, shard_size)
    else:
        re_img = resize_image(pil_img, shard_size*3, shard_size)
        
    # new_width, new_height = re_img.size

    # pad_width = custom_ceil(new_width, shard_size)
    # pad_height = custom_ceil(new_height, shard_size)
    # if new_width == new_height:
        # return re_img
    # else:
        # result = Image.new(re_img.mode, (pad_width, pad_height), background_color)
        # result.save('result.png')
        # result.paste(re_img, (0, 0))  # 将pil_img贴到results的(x,x)为止
        # result.save('result2.png')
        # return result
    return re_img


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    # prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
    prompt_chunks = [tokenizer(chunk, truncation=True, max_length=4096).input_ids for chunk in prompt.split('<image>')]

    input_ids = []
    input_ids.extend(prompt_chunks[0])
    for lst in prompt_chunks[1:]:
        input_ids.extend([image_token_index] + lst[1:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0]:] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
