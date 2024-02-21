"""
Apply the LoRA weights on top of a base model.

Usage:
python3 -m fastchat.model.apply_lora --base ~/model_weights/llama-7b --target ~/model_weights/baize-7b --lora project-baize/baize-lora-7B

Dependency:
pip3 install git+https://github.com/huggingface/peft.git@2822398fbe896f25d4dac5e468624dc5fd65a51b
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(os.path.split(curPath)[0])[0]
print(curPath, rootPath)
sys.path.append(rootPath)

sys.stdout = open(os.devnull, 'w')
from graphgpt.model import *

def apply_lora(base_model_path, target_model_path, lora_path):
    print(f"Loading the base model from {base_model_path}")
    # base = AutoModelForCausalLM.from_pretrained(
    #     base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    # )
    # base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    print('finish loading')

    print('start loading')
    base = GraphLlamaForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, use_cache=True,
                                                  low_cpu_mem_usage=True).cuda()
    print('finish loading')

    print(f"Loading the LoRA adapter from {lora_path}")

    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.float16,
    )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {target_model_path}")
    model.save_pretrained(target_model_path)
    base_tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--target_model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)

    args = parser.parse_args()

    apply_lora(args.base_model_path, args.target_model_path, args.lora_path)

# python graphgpt/model/apply_lora.py --base_model_path ./checkpoints/stage_ori/checkpoint-100 --target_model_path ./checkpoints/stage_final --lora_path ./checkpoints/stage_1/checkpoint-16800
