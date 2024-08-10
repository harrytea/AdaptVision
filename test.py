import argparse
import os
os.environ['CURL_CA_BUNDLE'] = ''
import sys

# from vqa import VQA
# from vqa_eval import VQAEval
# torchrun --nproc_per_node 8 --master_port 23111 evaluate_vqa_ours.py

sys.path.append("/data/wangyh/mllms/TGDoc_plus/AdaptVision_copy")
sys.path.insert(0, "/data/wangyh/mllms/TGDoc_plus/AdaptVision_copy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--LLaVA_model_path", type=str, default="/data/wangyh/mllms/TGDoc_plus/AdaptVision_copy/checkpoints/llava-7b-finetune")
    parser.add_argument("--LLaVA_model_base", type=str, default="/data/wangyh/mllms/LLaVA/checkpoints/vicuna-7b-v1.5")
    parser.add_argument("--image_file", default="asset/1_1mc.png", type=str)
    parser.add_argument("--question", default="describe this image", type=str)
    args = parser.parse_args()

    from LLaVA import LLaVA
    model = LLaVA(args.LLaVA_model_path, args.LLaVA_model_base, 'cuda')
    tokenizer = model.tokenizer

    print(model.generate(image=args.image_file, question=args.question))
