from PIL import Image

import re
import math
import torch
from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX



def sliding_window(img, stride):
    width, height = img.size
    img_all = img.resize((stride, stride), Image.LANCZOS)

    if width <= stride and height <= stride:
        return [img_all, img_all], [(0, 0), (1, 1)]

    split_images, split_index = [img_all], [(0, 0)]
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            split_images.append(img.crop((j, i, j+stride, i+stride)))
            split_index.append(((i+stride)//stride, (j+stride)//stride))

    return split_images, split_index


def process_grid_image(pil_img, shard_size, max_grid_num=3):
    width, height = pil_img.size

    max_side = max(width, height)
    scale = min((max_side + shard_size - 1) // shard_size, max_grid_num)
    max_length = shard_size * scale

    if height >= width:
        new_height = max_length
        new_width = max(int(max_length * width / height), 1)
        scale = min((new_width + shard_size - 1) // shard_size, max_grid_num)
        new_width = shard_size * scale
    else:
        new_width = max_length
        new_height = max(int(max_length * height / width), 1)
        scale = min((new_height + shard_size - 1) // shard_size, max_grid_num)
        new_height = shard_size * scale

    resized_img = pil_img.resize((new_width, new_height), Image.LANCZOS)  ######## max length
    return resized_img



def process_anyres_image(pil_img, shard_size, grid_pinpoints):
    """
        grid_pinpoints (list): [[384, 384], [384, 768], [384, 1152]...]
    """
    possible_grids = grid_pinpoints
    best_resolution = select_best_resolution(pil_img.size, shard_size, possible_grids)
    padded_image = resize_and_pad_image(pil_img, best_resolution)

    return padded_image



def get_anyres_image_grid_shape(image_size, shard_size, possible_grids):
    """
        - image_size: (200, 300)
        - shard_size: 336
        - possible_grids: '(1x1),...,(3x3)'
    """
    width, height = select_best_resolution(image_size, shard_size, possible_grids)
    return width // shard_size, height // shard_size


def resize_and_pad_image(image, target_resolution):
    """
        Resize and pad an image to a target resolution while maintaining aspect ratio.
        - target_resolution: tuple (768, 768)
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    # Determine which dimension (width or height) to fill
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        # Width will be filled completely
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        # Height will be filled completely
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    resized_image = image.resize((new_width, new_height))
    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def select_best_resolution(original_size, shard_size, possible_grids):
    """
    args
        - original_size: (200, 300)
        - shard_size: 336
        - possible_grids: '(1x1),...,(3x3)'

    return: (336, 336)
    """
    matches = re.findall(r"\((\d+)x(\d+)\)", possible_grids)
    range_start, range_end = tuple(map(int, matches[0])), tuple(map(int, matches[-1]))
    grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
    possible_grids = [[dim * shard_size for dim in pair] for pair in grid_pinpoints]

    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_grids:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)

        # Calculate effective and wasted resolutions
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit




def unpad_image(tensor, original_size, num_patch_width, num_patch_height):
    """
    Args:
        - tensor: torch.Size([4096, 36, 36])  # CxHxW 
        - original_size: [1000, 1200]
        - num_patch_width, num_patch_height: (3, 3)
    """
    C, H, W = tensor.shape
    orig_w, orig_h = original_size  # (W, H)

    # 计算 padding（假设对称 padding）
    cur_ar = W / H
    orig_ar = orig_w / orig_h

    if orig_ar > cur_ar:  # 高度方向有 padding
        scale = W / orig_w
        new_h = int(round(orig_h * scale))
        pad_top = max(0, (H - new_h) // 2)
        pad_bottom = H - new_h - pad_top
        pad_left = pad_right = 0
    elif orig_ar < cur_ar:  # 宽度方向有 padding
        scale = H / orig_h
        new_w = int(round(orig_w * scale))
        pad_left = max(0, (W - new_w) // 2)
        pad_right = W - new_w - pad_left
        pad_top = pad_bottom = 0
    else:
        pad_left = pad_right = pad_top = pad_bottom = 0

    # 在“含 padding”的整幅图上先划分网格
    h_edges = [(H * i) // num_patch_height for i in range(num_patch_height + 1)]
    w_edges = [(W * j) // num_patch_width for j in range(num_patch_width + 1)]

    # 有效（去 padding）区域
    up_h0, up_h1 = pad_top, H - pad_bottom
    up_w0, up_w1 = pad_left, W - pad_right

    patches = []
    for i in range(num_patch_height):
        for j in range(num_patch_width):
            h0, h1 = h_edges[i], h_edges[i + 1]
            w0, w1 = w_edges[j], w_edges[j + 1]
            hs = max(h0, up_h0)
            he = min(h1, up_h1)
            ws = max(w0, up_w0)
            we = min(w1, up_w1)
            patch = tensor[:, hs:he, ws:we]  # 若某块完全落在 padding，结果可能为空
            patches.append(patch)
    return patches


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX):
    prompt_chunks = [tokenizer(chunk, truncation=True, max_length=4096)['input_ids'] for chunk in prompt.split('<image>')]
    input_ids = prompt_chunks.pop(0)
    for lst in prompt_chunks:
        input_ids.extend([image_token_index] + lst[1:])

    return torch.tensor(input_ids, dtype=torch.long)

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
