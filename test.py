import os
import torch
import argparse
os.environ['CURL_CA_BUNDLE'] = ''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--LLaVA_model_path", type=str, default="/checkpoints/llava-7b-finetune")
    parser.add_argument("--image_file", default="asset/tree.png", type=str)
    parser.add_argument("--question", default="describe this image", type=str)
    args = parser.parse_args()

    from LLaVA import LLaVA
    # model = LLaVA(model_path=args.LLaVA_model_path, device='cpu', dtype=torch.float32)
    model = LLaVA(model_path=args.LLaVA_model_path, device='cuda', dtype=torch.float16)

    print(model.generate(image=args.image_file, question=args.question))
