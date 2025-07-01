import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

revisit_llava_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(revisit_llava_path)
sys.path.append(os.path.join(revisit_llava_path, 'data'))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, Conversation, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from chair_loader import CHAIRDataset

import warnings
warnings.filterwarnings(action='ignore')

import aiofiles
import asyncio

def set_random_seed(seed):
    print(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed) # Set the seed for NumPy operations
    torch.manual_seed(seed) # Set the seed for the CPU
    torch.cuda.manual_seed(seed) # Set the seed for the GPU
    torch.cuda.manual_seed_all(seed) # if you are using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def str2bool(v):
    if isinstance(v, bool):
        return v
    if 'true' in v.strip().lower():
        return True
    elif 'false' in v.strip().lower():
        return False
    elif 'none' in v.strip().lower():
        return None
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="CHAIR on LLaVA")
    parser.add_argument("--model_path", type=str, help="model")
    parser.add_argument("--model_base", type=str, default="llava")

    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--data_path", type=str, default="/root/data/coco/val2014/", help="data path")
    parser.add_argument("--anno_path", type=str, default="/root/data/coco/annotations/instances_val2014.json")
    parser.add_argument("--output_path", type=str, default=None, help="output path")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="num workers")

    parser.add_argument("--num_eval_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--exp_name", type=str, default='000')

    parser.add_argument("--do_sample", type=str2bool, default=False)
    parser.add_argument("--use_revisit", type=str2bool, default=True)
    parser.add_argument("--early_exit_layers", type=str, default="last", help="early exit layers")
    parser.add_argument("--relative_top", type=float, default=1e-5, help="relative top")

    args = parser.parse_known_args()[0]
    return args

async def main():
    args = parse_args()
    print("args: ", args)
    set_random_seed(args.seed)
    if args.output_path is not None:
        output_path = args.output_path
    else:
        output_path = os.path.join(revisit_llava_path, "output")
    output_file = os.path.join(output_path, f"{args.exp_name}.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print("output_file: ", output_file)

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    print("Model loaded")

    chair_dataset = CHAIRDataset(
        data_path=args.data_path,
        anno_path=args.anno_path,
        trans=image_processor,
        model=args.model_base
    )
    chair_loader = DataLoader(
        chair_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )

    qs =  "Please describe this image in detail."
    print("Dataset loaded")


    for batch_id, data in tqdm(enumerate(chair_loader), total=args.num_eval_samples):

        # early stop for debuggging purpose
        # if batch_id == 20:
            # break

        if batch_id == args.num_eval_samples:
            break
            
        img_id = data["image_id"]
        image_path = data["image_path"]
        image = data["image"]

        conv_out = Conversation(
            system="A chat between a curious human and an artificial intelligence assistant. "
                   "The assistant gives helpful, detailed, and polite answers to the human's questions.",
            roles=("USER", "ASSISTANT"),
            version="v1",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>",
        )
        qu_out = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv_out.append_message(conv_out.roles[0], qu_out)
        conv_out.append_message(conv_out.roles[1], None)
        prompt_out = conv_out.get_prompt()

        input_ids = tokenizer_image_token(prompt_out, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv_out.sep if conv_out.sep_style != SeparatorStyle.TWO else conv_out.sep2

        
        with torch.inference_mode():
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    images=image.unsqueeze(0).half().cuda(),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    do_sample=args.do_sample,
                    use_revisit=args.use_revisit,
                    early_exit_layers=args.early_exit_layers,
                    relative_top=args.relative_top,
                )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        print(f"[VQA for {args.exp_name}]")
        print(f"V: {image_path}")
        print(f"Q: {qs}")
        print(f"A: {outputs}")
        print(f"="*50)

        img_save = {}
        img_save["image_id"] = img_id.item()
        img_save["caption"] = outputs

        # dump metric file
        async with aiofiles.open(output_file, "a") as f:
            await f.write(json.dumps(img_save) + '\n')
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())