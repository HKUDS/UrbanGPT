import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(os.path.split(curPath)[0])[0]
print(curPath, rootPath)
sys.path.append(rootPath)

# sys.stdout = open(os.devnull, 'w')

from urbangpt.conversation import conv_templates, SeparatorStyle
from urbangpt.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from urbangpt.model import *
from urbangpt.model.utils import KeywordsStoppingCriteria
import json
import math
from peft import PeftModel
import copy

import requests
from PIL import Image
from io import BytesIO

from tqdm import tqdm
import json
import os.path as osp
import pickle

import ray

DEFAULT_STHIS_TOKEN = "<ST_HIS>"
DEFAULT_STPRE_TOKEN = "<ST_PRE>"
DEFAULT_ST_PATCH_TOKEN = "<ST_patch>"
DEFAULT_ST_START_TOKEN = "<ST_start>"
DEFAULT_ST_END_TOKEN = "<ST_end>"



def load_st(idx, instruct_item, st_data_all):

    sources = instruct_item

    region_start = int(sources["id"].split('_')[3])
    region_end = int(sources["id"].split('_')[4])
    i4data_all = int(sources["id"].split('_')[6])

    st_data_x = torch.Tensor(st_data_all[i4data_all]['data_x'])
    st_data_y = torch.Tensor(st_data_all[i4data_all]['data_y'])

    cur_token_len = 2

    return {
        'st_data_x': st_data_x,
        'st_data_y': st_data_y,
        'region_start': region_start,
        'region_end': region_end,
        'st_token_len': cur_token_len
    }


def load_prompting_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


# def prepare_query(instruct_item):


def run_eval(args, num_gpus):
    # split question file into num_gpus files
    prompt_file = load_prompting_file(args.prompting_file)
    prompt_file = prompt_file[args.start_id:args.end_id]
    # print('prompt_file_len', len(prompt_file))
    chunk_size = len(prompt_file) // num_gpus
    ans_handles = []
    split_list = list(range(args.start_id, args.end_id, chunk_size))
    idx_list = list(range(0, len(prompt_file), chunk_size))
    if len(split_list) == num_gpus:
        split_list.append(args.end_id)
        idx_list.append(len(prompt_file))
    elif len(split_list) == num_gpus + 1:
        split_list[-1] = args.end_id
        idx_list[-1] = len(prompt_file)
    else:
        raise ValueError('error in the number of list')

    print('idx_list', idx_list)

    if osp.exists(args.output_res_path) is False:
        os.makedirs(args.output_res_path, exist_ok=True)
        # os.mkdir(args.output_res_path)

    for idx in range(len(idx_list) - 1):
        start_idx = idx_list[idx]
        end_idx = idx_list[idx + 1]

        start_split = split_list[idx]
        end_split = split_list[idx + 1]
        ans_handles.append(
            eval_model.remote(
                args, prompt_file[start_idx:end_idx], start_split, end_split
            )
        )

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ray.get(ans_handle))

    # with open(args.output_res_path, "w") as ans_file:
    #     for line in ans_jsons:
    #         ans_file.write(json.dumps(line) + "\n")


@ray.remote(num_gpus=1)
@torch.inference_mode()
def eval_model(args, prompt_file, start_idx, end_idx):
    # load prompting file
    # prompt_file = load_prompting_file(args.prompting_file)

    # Model
    disable_torch_init()
    # model_name = os.path.expanduser(args.model_name)
    print('start loading')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print('finish loading')

    print('start loading')
    model = STLlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, use_cache=True,
                                                  low_cpu_mem_usage=True).cuda()
    model.set_st_tower()
    print('finish loading')

    use_st_start_end = getattr(model.config, "use_st_start_end", False)
    tokenizer.add_tokens([DEFAULT_ST_PATCH_TOKEN], special_tokens=True)
    if use_st_start_end:
        tokenizer.add_tokens([DEFAULT_ST_START_TOKEN, DEFAULT_ST_END_TOKEN], special_tokens=True)

    st_tower = model.get_model().st_tower


    st_config = st_tower.config
    st_config.st_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_ST_PATCH_TOKEN])[0]

    st_config.use_st_start_end = use_st_start_end
    if use_st_start_end:
        st_config.st_start_token, st_config.st_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_ST_START_TOKEN, DEFAULT_ST_END_TOKEN])


    res_data = []
    print(f'total: {len(prompt_file)}')
    with open(args.st_data_path, 'rb') as file:
        st_data_all = pickle.load(file)
    error_i = 0
    for idx, instruct_item in tqdm(enumerate(prompt_file)):
        st_dict = load_st(idx, instruct_item, st_data_all)
        st_token_len = st_dict['st_token_len']
        st_data_x = st_dict['st_data_x']
        st_data_y = st_dict['st_data_y']
        region_start = st_dict['region_start']
        region_end = st_dict['region_end']

        qs = instruct_item["conversations"][0]["value"]
        replace_token = DEFAULT_ST_PATCH_TOKEN * st_token_len
        replace_token = DEFAULT_ST_START_TOKEN + replace_token + DEFAULT_ST_END_TOKEN
        qs = qs.replace(DEFAULT_STHIS_TOKEN, replace_token)
        qs = qs.replace(DEFAULT_STPRE_TOKEN, replace_token)

        # if "v1" in args.model_name.lower():
        #     conv_mode = "stchat_v1"
        # else:
        #     raise ValueError('Don\'t support this model')
        conv_mode = "stchat_v1"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(
                conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)


        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                # st_data=st_data_x,
                st_data_x=st_data_x.cuda(),
                st_data_y=st_data_y.cuda(),
                region_start=region_start,
                region_end=region_end,
                do_sample=True,
                # do_sample=False,
                # temperature=0.2,
                temperature=0.01,
                # max_new_tokens=1024,
                max_new_tokens=256,
                stopping_criteria=[stopping_criteria])

            # Find the special tokens
            start_inx = torch.where(output_ids[0, :] == 32001)[0]
            end_inx = torch.where(output_ids[0, :] == 32002)[0]
            # Get hidden_states
            hidden_states = model.get_st_pre_res()
            hidden_states = torch.cat(hidden_states, dim=1)
            model.reset_st_pre_res()

            # Decode the token into the result
            batch_size = hidden_states.shape[0]
            feature_nums = 2
            st_pre_embs1 = hidden_states[:,
                           model.model.st_start_id0 + 1:model.model.st_start_id0 + feature_nums + 1,
                           :].detach().reshape(batch_size, -1, feature_nums, model.config.hidden_size)
            st_pre_out1 = model.relu(model.st_pred_linear_1(st_pre_embs1))

            if start_inx.shape[0] == 3:
                if hidden_states.shape[1] > start_inx[2] + 1 + feature_nums:
                    st_pre_embs2 = hidden_states[:, start_inx[2] + 1:start_inx[2] + 1 + feature_nums, :]
                else:
                    print('========error========')
                    error_i = error_i + 1
                    print(error_i)
                    print(hidden_states.shape, start_inx[2] + 1)
                    st_pre_embs2 = hidden_states[:, -(2+feature_nums):-2, :]
            else:
                print('========error========')
                error_i = error_i + 1
                print(error_i)
                st_pre_embs2 = hidden_states[:, -(2+feature_nums):-2, :]
            st_pre_embs2 = st_pre_embs2.reshape(batch_size, -1, feature_nums, model.config.hidden_size)
            st_pre_out2 = model.relu(model.st_pred_linear_3(st_pre_embs2))
            st_pre_final = model.st_pred_linear_2(torch.cat([st_pre_out1, st_pre_out2], dim=-1))
            st_pre_final = st_pre_final.transpose(-1, -2)
            st_pre_infolow = st_pre_final[:, :, :, 0].squeeze().detach().cpu().tolist()
            st_pre_outfolow = st_pre_final[:, :, :, 1].squeeze().detach().cpu().tolist()


        x_in, y_in = st_data_x[:, :, region_start:region_end, 0].squeeze().tolist(), st_data_y[0, :, region_start:region_end,
                                                                           0].squeeze().tolist()
        x_out, y_out = st_data_x[:, :, region_start:region_end, 1].squeeze().tolist(), st_data_y[0, :, region_start:region_end,
                                                                             1].squeeze().tolist()

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=False)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        res_data.append(
            {"id": instruct_item["id"], "res": outputs, "x_in": x_in, "x_out": x_out, "y_in": y_in, "y_out": y_out,
             "st_pre_infolow": st_pre_infolow, "st_pre_outfolow": st_pre_outfolow}.copy())
        with open(osp.join(args.output_res_path, 'arxiv_test_res_{}_{}.json'.format(start_idx, end_idx)), "w") as fout:
            json.dump(res_data, fout, indent=4)
    return res_data
    # with open(args.output_res_path, "w") as fout:
    #     json.dump(res_data, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    # parser.add_argument("--image-file", type=str, required=True)
    # parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--prompting_file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--st_data_path", type=str, default=None)

    parser.add_argument("--output_res_path", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=4)

    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=20567)

    args = parser.parse_args()

    # eval_model(args)

    ray.init()
    run_eval(args, args.num_gpus)

# protobuf             4.22.3