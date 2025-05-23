from huggingface_hub import snapshot_download
import argparse
import os

prerequisites_path = os.path.dirname(os.path.abspath(__file__))
def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA or QwenVL")
    parser.add_argument("--download_path", type=str, default=prerequisites_path, help="download path")
    parser.add_argument("--model", type=str, default="llava", help="model name")

def main():
    args = parse_args()
    args = parser.parse_args()
    if 'llava' in args.model:
        snapshot_download(repo_id="liuhaotian/llava-v1.5-7b", local_dir=os.path.join(args.download_path, "llava-v1.5-7b"))
    elif 'qwen' in args.model:
        snapshot_download(repo_id="Qwen/Qwen2.5-VL-7B-Instruct", local_dir=os.path.join(args.download_path, "Qwen2.5-VL-7B-Instruct"))
    else:
        raise ValueError("Invalid model name. Choose either 'llava' or 'qwen'.")

if __name__ == "__main__":
    main()