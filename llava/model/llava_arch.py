from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.constants import DEFAULT_IM_GLOBAL_TOKEN
from llava.constants import DEFAULT_IM_1_1_TOKEN, DEFAULT_IM_1_2_TOKEN, DEFAULT_IM_1_3_TOKEN
from llava.constants import DEFAULT_IM_2_1_TOKEN, DEFAULT_IM_2_2_TOKEN, DEFAULT_IM_2_3_TOKEN
from llava.constants import DEFAULT_IM_3_1_TOKEN, DEFAULT_IM_3_2_TOKEN, DEFAULT_IM_3_3_TOKEN

class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

    def get_vision_tower1(self):
        vision_tower1 = getattr(self, 'vision_tower1', None)
        if type(vision_tower1) is list:
            vision_tower1 = vision_tower1[0]
        return vision_tower1

    def get_vision_tower2(self):
        vision_tower2 = getattr(self, 'vision_tower2', None)
        if type(vision_tower2) is list:
            vision_tower2 = vision_tower2[0]
        return vision_tower2

    def initialize_vision_modules(self, model_args=None):
        vision_tower1 = build_vision_tower(model_args)
        vision_tower2 = build_vision_tower(model_args)

        self.vision_tower1 = vision_tower1
        self.vision_tower2 = vision_tower2

        # if model_args.pretrain_vision_tower is not None:
        #     vision_tower_weights = torch.load(model_args.pretrain_vision_tower, map_location='cpu')
        #     weights = {k[19:]: v for k, v in vision_tower_weights.items() if 'vision_tower' in k}
        #     is_load = False
        #     for k, v in vision_tower.named_parameters():
        #         if hasattr(v, "ds_id"):
        #             from deepspeed import zero
        #             with zero.GatheredParameters([v]):
        #                 v.data = weights[k].data             
        #             is_load = True
        #     if not is_load:
        #         self.vision_tower.load_state_dict(weights)

    def initialize_adapter_modules(self, model_args=None):
        self.mm_projector = nn.Linear(self.config.mm_hidden_size, self.config.hidden_size)
        self.down_vision = nn.Linear(self.config.hidden_size*4, self.config.hidden_size)
        if model_args.pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            self.down_vision.load_state_dict(get_w(mm_projector_weights, 'down_vision'))


    def update_config(self, model_args):
        self.config.mm_hidden_size = self.get_vision_tower1().hidden_size
        self.config.mm_vision_tower = model_args.vision_tower
        self.config.mm_vision_select_layer = model_args.mm_vision_select_layer
        self.config.mm_vision_select_feature = model_args.mm_vision_select_feature
        # pretrain
        self.config.pretrain_vision_tower = model_args.pretrain_vision_tower
        self.config.pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter


class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def encode_images(self, images):
        # # projector之前降维
        # image_features = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().mm_projector(image_features)
        # global_feature = image_features[0]
        # if image_features.size()[0] == 1:
        #     return [global_feature]
        # local_feature = image_features[1:]
        # bs, pn, hs = local_feature.shape
        # local_feature.view(bs, int(pn/4), int(hs*4)).size()
        # # local_feature = local_feature.permute(0, 2, 1)
        # local_feature = self.get_model().down_vision(local_feature)
        # # local_feature = local_feature.permute(0, 2, 1)
        # return [global_feature, local_feature]



        # 经过projector以后降维
        if images.size()[0] == 1:
            image_features = self.get_model().get_vision_tower1()(images)
        else:
            image_features1 = self.get_model().get_vision_tower1()(images[0].unsqueeze(0))
            image_features2 = self.get_model().get_vision_tower2()(images[1:])
            image_features = torch.cat((image_features1, image_features2), dim=0)
        image_features = self.get_model().mm_projector(image_features)
        global_feature = image_features[0]
        if image_features.size()[0] == 1:
            return [global_feature]
        local_feature = image_features[1:]
        bs, pn, hs = local_feature.shape
        local_feature = local_feature.view(bs, int(pn/4), int(hs*4))
        # local_feature = local_feature.permute(0, 2, 1)
        local_feature = self.get_model().down_vision(local_feature)
        # local_feature = local_feature.permute(0, 2, 1)
        return [global_feature, local_feature]

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower1 = self.get_model().get_vision_tower1()
        if vision_tower1 is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower1 is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels


        for idx, img_input in enumerate(images[0]):
            if img_input.dim() == 3:
                img_input = img_input.unsqueeze(0)
            images[0][idx] = self.encode_images(img_input)

        # image_features = self.encode_images(images[0][0])
        image_features = images[0]
        image_indexes = images[1]
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
            #     # multimodal LLM, but the current sample is not multimodal
            #     cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
            #     cur_input_embeds = cur_input_embeds + (0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
            #     new_input_embeds.append(cur_input_embeds)
            #     if labels is not None:
            #         new_labels.append(labels[batch_idx])
            #     cur_image_idx += 1
            #     continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                cur_image_indexes = image_indexes[cur_image_idx]
                # cur_image_features = cur_image_features.view(-1, self.config.hidden_size)
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    # global
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_image_indexes[0].unsqueeze(0)))
                    cur_new_input_embeds.append(cur_image_features[0])
                    # local ####################################################################
                    if len(cur_image_features) != 1:
                        for idx in range(cur_image_features[1].size()[0]):
                            cur_new_input_embeds.append(self.get_model().embed_tokens(cur_image_indexes[idx+1].unsqueeze(0)))
                            cur_new_input_embeds.append(cur_image_features[1][idx])
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        # global
                        image_label = cur_image_features[0].shape[0]
                        image_label = image_label + 1
                        # local
                        if len(cur_image_features) != 1:
                            image_label = image_label + cur_image_features[1].shape[0] * cur_image_features[1].shape[1]
                            image_label = image_label + cur_image_features[1].shape[0]
                        cur_new_labels.append(torch.full((image_label,), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start+1:image_token_start+2]) # +1, +2? ## 以前是+0， +1
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    # global
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_image_indexes[0].unsqueeze(0)))  ###########################
                    cur_new_input_embeds.append(cur_image_features[0])  ###########################
                    # local ####################################################################
                    if len(cur_image_features) != 1:
                        for idx in range(cur_image_features[1].size()[0]):
                            cur_new_input_embeds.append(self.get_model().embed_tokens(cur_image_indexes[idx+1].unsqueeze(0)))  ###########################
                            cur_new_input_embeds.append(cur_image_features[1][idx])  ###########################
                    # cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        # global
                        image_label = cur_image_features[0].shape[0]  ###########################
                        image_label = image_label + 1  ###########################
                        # local
                        if len(cur_image_features) != 1:
                            image_label = image_label + cur_image_features[1].shape[0] * cur_image_features[1].shape[1]  ###########################
                            image_label = image_label + cur_image_features[1].shape[0]  ###########################
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

        # print("=======================================================================================")
        return None, attention_mask, past_key_values, new_input_embeds, new_labels


    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_GLOBAL_TOKEN,
                                                   DEFAULT_IM_1_1_TOKEN, DEFAULT_IM_1_2_TOKEN, DEFAULT_IM_1_3_TOKEN,
                                                   DEFAULT_IM_2_1_TOKEN, DEFAULT_IM_2_2_TOKEN, DEFAULT_IM_2_3_TOKEN,
                                                   DEFAULT_IM_3_1_TOKEN, DEFAULT_IM_3_2_TOKEN, DEFAULT_IM_3_3_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))  # modify the embeded matrix

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data  # LlamaModel.embed_tokens=nn.Embedding(vocab_size, hidden_size)
                output_embeddings = self.get_output_embeddings().weight.data  # LlamaForCausalLM.lm_head=nn.Linear(hidden_size, vocab_size)

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 12
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
