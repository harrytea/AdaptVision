from transformers import AutoTokenizer, AutoConfig
import torch

from llava import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from transformers import StoppingCriteria
from llava.mm_utils import tokenizer_image_token
from llava.mm_utils import expand2square, sliding_window


from llava.constants import DEFAULT_IM_GLOBAL_TOKEN
from llava.constants import DEFAULT_IM_1_1_TOKEN, DEFAULT_IM_1_2_TOKEN, DEFAULT_IM_1_3_TOKEN
from llava.constants import DEFAULT_IM_2_1_TOKEN, DEFAULT_IM_2_2_TOKEN, DEFAULT_IM_2_3_TOKEN
from llava.constants import DEFAULT_IM_3_1_TOKEN, DEFAULT_IM_3_2_TOKEN, DEFAULT_IM_3_3_TOKEN

from PIL import Image
# from ..process import pad_image, resize_image
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<img>"
DEFAULT_IM_END_TOKEN = "</img>"
IMAGE_TOKEN_INDEX = -200
from peft import PeftModel, PeftConfig     
import os

def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }
    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False
        

class LLaVA:
    def __init__(self, model_path, model_base, device) -> None:
        config = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_base)
        patch_config(model_path)
        model = LlavaLlamaForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, config=config, low_cpu_mem_usage=True).to(device)

        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional LLaVA weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')

            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}


        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_GLOBAL_TOKEN,
                                DEFAULT_IM_1_1_TOKEN, DEFAULT_IM_1_2_TOKEN, DEFAULT_IM_1_3_TOKEN,
                                DEFAULT_IM_2_1_TOKEN, DEFAULT_IM_2_2_TOKEN, DEFAULT_IM_2_3_TOKEN,
                                DEFAULT_IM_3_1_TOKEN, DEFAULT_IM_3_2_TOKEN, DEFAULT_IM_3_3_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        

        # vision_encoder  ||| vision, adapter, llm == false, true, true
        config.image_size=336
        model.get_model().initialize_vision_modules(model_args=config)
        vision_tower1 = model.get_model().get_vision_tower1()
        vision_tower1.to(dtype=torch.float16, device='cuda')
        image_processor = vision_tower1.image_processor
        vision_tower2 = model.get_model().get_vision_tower2()
        vision_tower2.to(dtype=torch.float16, device='cuda')
        # mm_adapter
        model.get_model().initialize_adapter_modules(model_args=config)
        model.get_model().mm_projector.to(dtype=torch.float16, device='cuda')
        model.get_model().down_vision.to(dtype=torch.float16, device='cuda')
        # model.get_model().before_llm.to(dtype=torch.float16, device='cuda')
        # model.load_state_dict(non_lora_trainables, strict=False)  # 主要是tokenlizer加载一下

        # non lora 使用这个
        weights1 = torch.load(os.path.join(model_path, "pytorch_model-00001-of-00002.bin"), map_location='cpu')
        weights2 = torch.load(os.path.join(model_path, "pytorch_model-00002-of-00002.bin"), map_location='cpu')
        model.load_state_dict(weights1, strict=False)  # 主要是tokenlizer加载一下
        model.load_state_dict(weights2, strict=False)  # 主要是tokenlizer加载一下
        # def get_w(weights, keyword):
            # return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        # model.get_model().mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)


        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            model.load_state_dict(non_lora_trainables, strict=False)  # 主要是tokenlizer加载一下


        for n,p in model.named_parameters():
            if p.dtype == torch.float32:
                print(n)

        # model.half()
        print('Loading LoRA weights...')
        # model = PeftModel.from_pretrained(model, model_path)
        model.cuda()
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device


    def generate(self, image, question, max_new_toekns=256, name='resize'):
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv = conv_templates['v1'].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)  # add
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # process image
        # image_size = self.model.base_model.model.base_model.vision_tower.config.image_size
        image = Image.open(image).convert('RGB')
        # pad
        process_size = self.image_processor.crop_size['height']
        image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean), process_size)
        # image = np.array(image.resize((448, 448)))
        windows_img, windows_index = sliding_window(image, stride=process_size)
        

        for idx, win_index in enumerate(windows_index):
            joined_str = ",".join(str(num) for num in win_index)
            index_str = f"<{joined_str}>"
            tensor_index = self.tokenizer.encode(index_str)[1]
            windows_index[idx] = torch.tensor(tensor_index, dtype=torch.long).to(self.device)
        
        
        image_concat = []
        for img in windows_img:
            img = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].half()
            image_concat.append(img)
        image = torch.stack(image_concat, dim=0)
        image = image.half()
        image_tensor = ([image.to(self.device)], [windows_index])


        # if name == "pad":
            # image = pad_image(image, (image_size,image_size))
        # elif name == "resize":
            # image = resize_image(image, (image_size,image_size))
        # image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        # image_tensor = image_tensor.half()

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                images=image_tensor,
                do_sample=False,  # 加入随机性
                temperature=0.2,  # 0.2 // 0.9
                max_new_tokens=max_new_toekns,)
                # stopping_criteria=[stopping_criteria])
            input_token_len = input_ids.shape[1]
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

        return outputs

