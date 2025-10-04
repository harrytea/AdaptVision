import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import argparse
os.environ['CURL_CA_BUNDLE'] = ''



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--LLaVA_model_path", type=str, default="/opt/MultimodalOCR/MultimodalOCR/model/AdaptVision/checkpoints/llava-7b-grid3-finetune")
    parser.add_argument("--image_file", default="/opt/MultimodalOCR/MultimodalOCR/model/AdaptVision/images/arch.png", type=str)
    parser.add_argument("--question", default="describe this image", type=str)
    args = parser.parse_args()

    from AdaptVision import AdaptVisionHandler
    # model = LLaVA(model_path=args.LLaVA_model_path, device='cpu', dtype=torch.float32)
    model = AdaptVisionHandler()
    model.initialize_llm(model_path=args.LLaVA_model_path, device='cuda', dtype=torch.float16)

    print(model.generate(image=args.image_file, question=args.question))
