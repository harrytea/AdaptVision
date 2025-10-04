from abc import ABC, abstractmethod
import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .multimodal_encoder.builder import build_vision_tower

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.constants import DEFAULT_IM_GLOBAL_TOKEN
from llava.constants import DEFAULT_IM_1_1_TOKEN, DEFAULT_IM_1_2_TOKEN, DEFAULT_IM_1_3_TOKEN, DEFAULT_IM_1_4_TOKEN
from llava.constants import DEFAULT_IM_2_1_TOKEN, DEFAULT_IM_2_2_TOKEN, DEFAULT_IM_2_3_TOKEN, DEFAULT_IM_2_4_TOKEN
from llava.constants import DEFAULT_IM_3_1_TOKEN, DEFAULT_IM_3_2_TOKEN, DEFAULT_IM_3_3_TOKEN, DEFAULT_IM_3_4_TOKEN
from llava.constants import DEFAULT_IM_4_1_TOKEN, DEFAULT_IM_4_2_TOKEN, DEFAULT_IM_4_3_TOKEN, DEFAULT_IM_4_4_TOKEN
from llava.mm_utils import get_anyres_image_grid_shape, unpad_image


class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

    def get_vision_global(self):
        return self.vision_global

    def get_vision_local(self):
        return self.vision_local

    def initialize_vision_modules(self, model_args=None):
        self.vision_global = build_vision_tower(model_args)
        self.vision_local = build_vision_tower(model_args)

    def initialize_adapter_modules(self):
        self.mm_projector = nn.Linear(self.config.mm_hidden_size, self.config.hidden_size)
        self.down_vision = nn.Linear(self.config.hidden_size*4, self.config.hidden_size)

    def update_config(self, model_args):
        self.config.mm_hidden_size = self.get_vision_global().hidden_size
        self.config.mm_vision_tower = model_args.vision_tower
        self.config.mm_vision_select_layer = model_args.mm_vision_select_layer
        self.config.mm_vision_select_feature = model_args.mm_vision_select_feature
        # pretrain
        self.config.pretrain_vision_tower = model_args.pretrain_vision_tower
        self.config.image_aspect_ratio = model_args.image_aspect_ratio
        self.config.image_grid_pinpoints = model_args.image_grid_pinpoints
        self.config.max_grid_num = model_args.max_grid_num
        self.config.use_pos_token = model_args.use_pos_token

class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    #--------------------------------------------------
    def encode_images_batch(self, images):
        batch_size = len(images)
        index_list = [len(img)-1 for img in images]  # 减去全局图像

        # 图像编码
        global_images = torch.stack([img[0] for img in images]) # [8, 3, 336, 336]
        local_images = torch.cat([img[1:] for img in images])  # [63, 3, 336, 336]
        image_global = self.get_model().get_vision_global()(global_images)  # [8, n_tokens, hidden_size]
        image_local = self.get_model().get_vision_local()(local_images)    # [63, n_tokens, hidden_size]

        # projector
        global_features = self.get_model().mm_projector(image_global)
        local_features = self.get_model().mm_projector(image_local)
        # down
        bs_local, seq_len, hidden_size = local_features.shape
        #### 针对 siglip2, 729 不能被 4 整除, 加入 padding 操作
        if seq_len % 4 != 0:
            pad_len = 4 - (seq_len % 4)
            local_features = F.pad(local_features, (0, 0, 0, pad_len), "constant", 0)
            seq_len += pad_len
        #### 针对 siglip2, 729 不能被 4 整除, 加入 padding 操作
        local_features = local_features.view(bs_local, int(seq_len // 4), int(hidden_size * 4))
        local_features = self.get_model().down_vision(local_features)

        # 重塑局部特征，计算累积索引
        cumulative_indices = [0] + list(itertools.accumulate(index_list))
        # images_return = []
        # for i in range(batch_size):
        #     start_idx, end_idx = cumulative_indices[i], cumulative_indices[i+1]
        #     images_return.append([global_features[i], local_features[start_idx:end_idx]])
        images_return = [
            [global_features[i], local_features[cumulative_indices[i]:cumulative_indices[i+1]]]
            for i in range(batch_size)
        ]
        return images_return
    #--------------------------------------------------


    def prepare_inputs_labels_for_multimodal(self, input_ids, attention_mask, past_key_values, labels, images, image_sizes):
        vision_global = self.get_model().get_vision_global()
        if vision_global is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_global is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        #--------------------------------------------------
        images[0] = self.encode_images_batch(images[0])
        #--------------------------------------------------

        image_features, image_indexes = images[0], images[1]
        if self.config.image_aspect_ratio == "anyres":
            for image_idx, image_feature in enumerate(image_features):
                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                    image_sizes[image_idx], self.get_model().get_vision_global().image_size, self.config.image_grid_pinpoints
                )
                num_patches_per_side = self.get_model().get_vision_global().num_patches_per_side
                down = 4 // 2 # TODO Fix
                num_patches_per_side = num_patches_per_side // down
                #### 针对 siglip2 的特殊处理
                while num_patches_per_side * num_patches_per_side < image_feature[1].shape[1]:
                    num_patches_per_side += 1  # 增加 num_patches_per_side, 直到面积 >= 183
                target_seq_len = num_patches_per_side * num_patches_per_side
                pad_len = target_seq_len - image_feature[1].shape[1]  # 需要填充的长度
                if pad_len > 0:
                    image_feature[1] = F.pad(image_feature[1], (0, 0, 0, pad_len), "constant", 0)
                image_feature[1] = image_feature[1].view(num_patch_width, num_patch_height, num_patches_per_side, num_patches_per_side, -1)
                #### 针对 siglip2 的特殊处理
                # image_feature[1] = image_feature[1].view(num_patch_width, num_patch_height, num_patches_per_side, num_patches_per_side, -1)
                image_feature[1] = image_feature[1].permute(4, 0, 2, 1, 3).contiguous()
                image_feature[1] = image_feature[1].flatten(1, 2).flatten(2, 3)
                image_feature[1] = unpad_image(image_feature[1], image_sizes[image_idx], num_patch_width, num_patch_height)
                image_feature[1] = [patch.permute(1, 2, 0).reshape(-1, patch.shape[0]) for patch in image_feature[1]]
        elif self.config.image_aspect_ratio == "grid":
            for image_idx, image_feature in enumerate(image_features):
                image_feature[1] = list(torch.unbind(image_feature[1], dim=0))

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                cur_image_indexes = image_indexes[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())  # before <image>
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))  # <image>
                    # global
                    if self.config.use_pos_token:
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_image_indexes[0].unsqueeze(0)))  # global pos token
                    cur_new_input_embeds.append(cur_image_features[0])  # global feature
                    # local ####################################################################
                    for idx in range(len(cur_image_features[1])):
                        if self.config.use_pos_token:
                            cur_new_input_embeds.append(self.get_model().embed_tokens(cur_image_indexes[idx+1].unsqueeze(0)))  # local pos token
                        cur_new_input_embeds.append(cur_image_features[1][idx])  # local feature
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))  # </image>
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        # global
                        image_label = cur_image_features[0].shape[0]  # global image label
                        if self.config.use_pos_token:
                            image_label = image_label + 1  # global pos label
                        # local ####################################################################
                        image_label = image_label + sum(patch.shape[0] for patch in cur_image_features[1])  # local image label
                        if self.config.use_pos_token:
                            image_label = image_label + len(cur_image_features[1])  # local pos label
                        cur_new_labels.append(torch.full((image_label,), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start+1:image_token_start+2]) # +1, +2? ## 以前是+0， +1
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    # global
                    if self.config.use_pos_token:
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_image_indexes[0].unsqueeze(0)))  ###########################
                    cur_new_input_embeds.append(cur_image_features[0])  ###########################
                    # local ####################################################################
                    for idx in range(len(cur_image_features[1])):
                        if self.config.use_pos_token:
                            cur_new_input_embeds.append(self.get_model().embed_tokens(cur_image_indexes[idx+1].unsqueeze(0)))  ###########################
                        cur_new_input_embeds.append(cur_image_features[1][idx])  ###########################
                    # cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        # global
                        image_label = cur_image_features[0].shape[0]  ###########################
                        if self.config.use_pos_token:
                            image_label = image_label + 1  ###########################
                        image_label = image_label + sum(patch.shape[0] for patch in cur_image_features[1])  ###########################
                        if self.config.use_pos_token:
                            image_label = image_label + len(cur_image_features[1])  ###########################
                        cur_new_labels.append(torch.full((image_label,), IGNORE_INDEX, device=labels.device, dtype=labels.dtype)) ###########################
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1

                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)    # finetune
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len-cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else: 
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels


    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_GLOBAL_TOKEN,
                                                   DEFAULT_IM_1_1_TOKEN, DEFAULT_IM_1_2_TOKEN, DEFAULT_IM_1_3_TOKEN, DEFAULT_IM_1_4_TOKEN,
                                                   DEFAULT_IM_2_1_TOKEN, DEFAULT_IM_2_2_TOKEN, DEFAULT_IM_2_3_TOKEN, DEFAULT_IM_2_4_TOKEN,
                                                   DEFAULT_IM_3_1_TOKEN, DEFAULT_IM_3_2_TOKEN, DEFAULT_IM_3_3_TOKEN, DEFAULT_IM_3_4_TOKEN,
                                                   DEFAULT_IM_4_1_TOKEN, DEFAULT_IM_4_2_TOKEN, DEFAULT_IM_4_3_TOKEN, DEFAULT_IM_4_4_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))  # modify the embeded matrix

            if num_new_tokens > 0:  # initialize the new tokens
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg 

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False