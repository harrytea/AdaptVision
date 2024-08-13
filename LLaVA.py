import os
import torch

from llava import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.mm_utils import expand2square, sliding_window
from transformers import AutoTokenizer, AutoConfig

from PIL import Image
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<img>"
DEFAULT_IM_END_TOKEN = "</img>"
IMAGE_TOKEN_INDEX = -200
        

class LLaVA:
    def __init__(self, model_path, device, dtype):
        config = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = LlavaLlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype, config=config).to(device)
        model.get_model().initialize_vision_modules(model_args=config)
        vision_global = model.get_model().get_vision_global().to(dtype=dtype, device=device)
        vision_local = model.get_model().get_vision_local().to(dtype=dtype, device=device)

        model.get_model().initialize_adapter_modules()
        model.get_model().mm_projector.to(dtype=dtype, device=device)
        model.get_model().down_vision.to(dtype=dtype, device=device)
        image_processor = vision_global.image_processor


        weights = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location='cpu')
        model.load_state_dict(weights, strict=False)  # 主要是tokenlizer加载一下

        for n,p in model.named_parameters():
            if p.dtype == torch.float32:
                print(n)

        model.to(dtype=dtype, device=device)
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device
        self.dtype = dtype


    def generate(self, image, question, max_new_toekns=256):
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv = conv_templates['v1'].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)  # add
        prompt = conv.get_prompt()
        print(prompt)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX).unsqueeze(0).to(self.device)


        image = Image.open(image).convert('RGB')
        crop_height = self.image_processor.crop_size['height']
        image = expand2square(image, crop_height)
        windows_img, windows_index = sliding_window(image, stride=crop_height)
    
        image_concat = []
        for img in windows_img:
            img = self.image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0]
            image_concat.append(img)
        image = torch.stack(image_concat, dim=0)
        image = torch.as_tensor(image, dtype=self.dtype)

        # position index
        for idx, win_index in enumerate(windows_index):
            joined_str = f"<{win_index[0]},{win_index[1]}>"
            tensor_index = self.tokenizer.encode(joined_str)[1]
            windows_index[idx] = torch.tensor(tensor_index, dtype=torch.long).to(self.device)

        image_tensor = ([image.to(self.device)], [windows_index])
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                images=image_tensor,
                do_sample=False,  # 加入随机性
                temperature=0.2,  # 0.2 // 0.9
                max_new_tokens=max_new_toekns
            )
            input_token_len = input_ids.shape[1]
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

        return outputs

