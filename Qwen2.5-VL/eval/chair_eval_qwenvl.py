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

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
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
    parser = argparse.ArgumentParser(description="CHAIR on QwenVL")
    parser.add_argument("--model_path", type=str, help="model")
    parser.add_argument("--model_base", type=str, default="qwenvl")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1)

    parser.add_argument("--data_path", type=str, default="/root/data/coco/val2014/", help="data path")
    parser.add_argument("--anno_path", type=str, default="/root/data/coco/annotations/instances_val2014.json")
    parser.add_argument("--output_path", type=str, default=None, help="output path")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="num workers")

    parser.add_argument("--num_eval_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--experiment_index", type=int, default=0)

    parser.add_argument("--do_sample", action=str2bool, help="sample")
    parser.add_argument("--use_revisit", action=str2bool, help="use revisit")
    parser.add_argument("--early_exit_layers", type=str, default="all", help="early exit layers")
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
        output_path = os.path.join(revisit_llava_path, "output", "chair")
    os.makedirs(output_path, exist_ok=True)
    print("output_path: ", output_path)
    output_file = os.path.join(output_path, f"exp_{args.experiment_index}.jsonl")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    print("Model loaded")

    chair_dataset = CHAIRDataset(
        data_path=args.data_path,
        anno_path=args.anno_path,
        trans=None,
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
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": qs},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )



    for batch_id, data in tqdm(enumerate(chair_loader), total=args.num_eval_samples):

        # early stop for debuggging purpose
        # if batch_id == 20:
            # break

        if batch_id == args.num_eval_samples:
            break
            
        image_id = data['image_id']
        image_path = data["image_path"]

        inputs = processor(
            text=[text for i in range(len(image_path))],
            images=[Image.open(img) for img in image_path],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        
        with torch.inference_mode():
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    repetition_penalty=args.repetition_penalty,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    use_revisit=args.use_revisit,
                    early_exit_layers=args.early_exit_layers,
                    relative_top=args.relative_top,
                )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


        print(f"[VQA for ReVisiT]")
        print(f"V: {image_path}")
        print(f"Q: {qs}")
        print(f"A: {outputs}")
        print(f"="*50)

        img_save = {}
        img_save["image_id"] = image_id.item()
        img_save["caption"] = outputs

        # dump metric file
        async with aiofiles.open(output_file, "a") as f:
            await f.write(json.dumps(img_save) + '\n')
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())