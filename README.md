# UrbanGPT: Spatio-Temporal Large Language Models

A pytorch implementation for the paper: [UrbanGPT: Spatio-Temporal Large Language Models]<br />  

[Zhonghang Li](https://scholar.google.com/citations?user=__9uvQkAAAAJ), [Lianghao Xia](https://akaxlh.github.io/), [Jiabin Tang](https://tjb-tech.github.io/), [Yong Xu](https://scholar.google.com/citations?user=1hx5iwEAAAAJ), [Lei Shi](https://harryshil.github.io/), [Long Xia](https://scholar.google.com/citations?user=NRwerBAAAAAJ), [Dawei Yin](https://www.yindawei.com/), [Chao Huang](https://sites.google.com/view/chaoh)* (*Correspondence)<br />  

**[Data Intelligence Lab](https://sites.google.com/view/chaoh/home)@[University of Hong Kong](https://www.hku.hk/)**, [South China University of Technology](https://www.scut.edu.cn/en/), Baidu Inc  

-----

<a href='https://urban-gpt.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://github.com/HKUDS/UrbanGPT'><img src='https://img.shields.io/badge/Demo-Page-purple'></a> 
<a href='https://arxiv.org/abs/2403.00813'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a> 
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=4BIbQt-EIAM)
 • 🌐 <a href="https://zhuanlan.zhihu.com/p/684785925" target="_blank">中文博客</a>

This repository hosts the code, data, and model weights of **UrbanGPT**.

-----
## 🎉 News 

- [x] 🚀🔥 [2024.05] 🎯🎯📢📢 Exciting News! We are thrilled to announce that our 🌟UrbanGPT🌟 has been accepted by KDD'2024! 🎉🎉🎉 Thanks to all the team members 🤗

🎯🎯📢📢 We upload the **models** and **data** used in our UrbanGPT on 🤗 **Huggingface**. We highly recommend referring to the table below for further details: 

| 🤗 Huggingface Address                                        | 🎯 Description                                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [https://huggingface.co/bjdwh/UrbanGPT](https://huggingface.co/bjdwh/UrbanGPT/tree/main) | It's the checkpoint of our UrbanGPT based on Vicuna-7B-v1.5-16k tuned on instruction data [train-data](https://huggingface.co/datasets/bjdwh/ST_data_urbangpt/tree/main/train_data) |
| [https://huggingface.co/datasets/bjdwh/ST_data_urbangpt](https://huggingface.co/datasets/bjdwh/ST_data_urbangpt) | We release a portion of the dataset for evaluation. |

- [x] [2023.02.23] 🚀🚀 Release the code of UrbanGPT.
- [x] [2023.02.29] Add video.
- [x] [2023.03.05] Release the full paper.
- [x] [2023.03.11] Upload the new checkpoint of our UrbanGPT.

## 👉 TODO 
- [ ] Release more st dataset.
- [ ] Release instruction generation codes.
- [ ] Release baselines codes.
- [ ] ...


-----------

## Introduction

<p style="text-align: justify">
In this work, we present a spatio-temporal large language model that can exhibit exceptional generalization capabilities across a wide range of downstream urban tasks. 
To achieve this objective, we present the UrbanGPT, which seamlessly integrates a spatio-temporal dependency encoder with the instruction-tuning paradigm. 
This integration enables large language models (LLMs) to comprehend the complex inter-dependencies across time and space, facilitating more comprehensive and accurate predictions under data scarcity. 
Extensive experimental findings highlight the potential of building LLMs for spatio-temporal learning, particularly in zero-shot scenarios.
</p>

![The detailed framework of the proposed UrbanGPT.](https://github.com/urban-gpt/urban-gpt.github.io/blob/main/images/urbangpt_framework.png)


### Demo Video
https://github.com/HKUDS/UrbanGPT/assets/90381931/9cd094b4-8fa3-486f-890d-631a08b19b4a

-----------
<span id='Usage'/>

## Getting Started

<span id='all_catelogue'/>

### Table of Contents:
* <a href='#Code Structure'>1. Code Structure</a>
* <a href='#Environment'>2. Environment </a>
* <a href='#Training UrbanGPT'>3. Training UrbanGPT </a>
  * <a href='#Prepare Pre-trained Checkpoint'>3.1. Prepare Pre-trained Checkpoint</a>
  * <a href='#Instruction Tuning'>3.2. Instruction Tuning</a>
* <a href='#Evaluating UrbanGPT'>4. Evaluating UrbanGPT</a>
  * <a href='#Preparing Checkpoints and Data'>4.1. Preparing Checkpoints and Data</a>
  * <a href='#Running Evaluation'>4.2. Running Evaluation</a>

****


<span id='Code Structure'/>

### 1. Code Structure <a href='#all_catelogue'>[Back to Top]</a>

```
.
|   README.md
|   urbangpt_eval.sh
|   urbangpt_train.sh
|   
+---checkpoints
|   \---st_encoder
|           pretrain_stencoder.pth
|           
+---playground
|   |   inspect_conv.py
|   |   
|   +---test_embedding
|   |       README.md
|   |       test_classification.py
|   |       test_semantic_search.py
|   |       test_sentence_similarity.py
|   |       
|   \---test_openai_api
|           anthropic_api.py
|           openai_api.py
|           
+---tests
|       test_openai_curl.sh
|       test_openai_langchain.py
|       test_openai_sdk.py
|       
\---urbangpt
    |   constants.py
    |   conversation.py
    |   utils.py
    |   __init__.py
    |   
    +---eval
    |   |   run_urbangpt.py                     # evaluation
    |   |   run_vicuna.py
    |   |   
    |   \---script
    |           run_model_qa.yaml
    |           
    +---model
    |   |   apply_delta.py
    |   |   apply_lora.py
    |   |   builder.py
    |   |   compression.py
    |   |   convert_fp16.py
    |   |   make_delta.py
    |   |   model_adapter.py
    |   |   model_registry.py
    |   |   monkey_patch_non_inplace.py
    |   |   STLlama.py                          # model
    |   |   utils.py
    |   |   __init__.py
    |   |   
    |   \---st_layers
    |           args.py
    |           ST_Encoder.conf
    |           ST_Encoder.py                   # ST-Encoder
    |           __init__.py
    |           
    +---protocol
    |       openai_api_protocol.py
    |       
    +---serve
    |   |   api_provider.py
    |   |   bard_worker.py
    |   |   cacheflow_worker.py
    |   |   cli.py
    |   |   controller.py
    |   |   controller_graph.py
    |   |   gradio_block_arena_anony.py
    |   |   gradio_block_arena_named.py
    |   |   gradio_css.py
    |   |   gradio_patch.py
    |   |   gradio_web_server.py
    |   |   gradio_web_server_graph.py
    |   |   gradio_web_server_multi.py
    |   |   huggingface_api.py
    |   |   inference.py
    |   |   model_worker.py
    |   |   model_worker_graph.py
    |   |   openai_api_server.py
    |   |   register_worker.py
    |   |   test_message.py
    |   |   test_throughput.py
    |   |   __init__.py
    |   |   
    |   +---examples
    |   |       extreme_ironing.jpg
    |   |       waterview.jpg
    |   |       
    |   +---gateway
    |   |       nginx.conf
    |   |       README.md
    |   |       
    |   \---monitor
    |           basic_stats.py
    |           clean_battle_data.py
    |           elo_analysis.py
    |           hf_space_leaderboard_app.py
    |           monitor.py
    |           
    \---train
            llama2_flash_attn_monkey_patch.py
            llama_flash_attn_monkey_patch.py
            stchat_trainer.py
            train_lora.py
            train_mem.py
            train_st.py                         # train
            
```


<span id='Environment'/>

### 2.Environment <a href='#all_catelogue'>[Back to Top]</a>
Please first clone the repo and install the required environment, which can be done by running the following commands:
```shell
conda create -n urbangpt python=3.9.13

conda activate urbangpt

# Torch with CUDA 11.7
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# To support vicuna base model
pip3 install "fschat[model_worker,webui]"

# To install pyg and pyg-relevant packages
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

# Clone our UrabnGPT or download it
git clone https://github.com/HKUDS/UrbanGPT.git
cd UrbanGPT

# Install required libraries
# (The recommendation is to install separately using the following method)
pip install deepspeed
pip install ray
pip install einops
pip install wandb
# （There is a version compatibility issue between "flash-attn" and "transformers". Please refer to the flash-attn [GitHub URL](https://github.com/Dao-AILab/flash-attention) for more information.）
pip install flash-attn==2.3.5  # or download from (https://github.com/Dao-AILab/flash-attention/releases, e.g. flash_attn-2.3.5+cu117torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl)
pip install transformers==4.34.0

# （or you can install according to the requirements file.）
pip install -r requirements.txt
```

<span id='Training UrbanGPT'/>

### 3. Training UrbanGPT <a href='#all_catelogue'>[Back to Top]</a>

<span id='Prepare Pre-trained Checkpoint'/>

#### 3.1. Preparing Pre-trained Checkpoint  <a href='#all_catelogue'>[Back to Top]</a>
UrabnGPT is trained based on following excellent existing models.
Please follow the instructions to prepare the checkpoints.

- `Vicuna`:
  Prepare our base model Vicuna, which is an instruction-tuned chatbot and base model in our implementation. Please download its weights [here](https://github.com/lm-sys/FastChat#model-weights). We generally utilize v1.5 and v1.5-16k model with 7B parameters. You should update the 'config.json' of vicuna, for example, the 'config.json' in v1.5-16k can be found in [config.json](https://huggingface.co/datasets/bjdwh/checkpoints/blob/main/train_config/config.json)

- `Spatio-temporal Encoder`:
  We employ a simple TCNs-based spatio-temporal encoder to encode the spatio-temporal dependencies. The weights of [st_encoder](./checkpoints/st_encoder/pretrain_stencoder.pth) are pre-trained through a typical multi-step spatio-temporal prediction task.

- `Spatio-temporal Train Data`:
  We utilize pre-training data consisting of New York City's taxi, bike, and crime data, including spatio-temporal statistics, recorded timestamps, and information about regional points of interest (POIs). These data are organized in [train_data](https://huggingface.co/datasets/bjdwh/ST_data_urbangpt/tree/main/train_data). Please download it and put it at ./UrbanGPT/ST_data_urbangpt/train_data

<span id='Instruction Tuning'/>

#### 3.2. Instruction Tuning <a href='#all_catelogue'>[Back to Top]</a>

* **Start tuning:** After the aforementioned steps, you could start the instruction tuning by filling blanks at [urbangpt_train.sh](urbangpt_train.sh). There is an example as below: 

```shell
# to fill in the following path to run our UrbanGPT!
model_path=./checkpoints/vicuna-7b-v1.5-16k
instruct_ds=./ST_data_urbangpt/train_data/multi_NYC.json
st_data_path=./ST_data_urbangpt/train_data/multi_NYC_pkl.pkl
pretra_ste=ST_Encoder
output_model=./checkpoints/UrbanGPT

wandb offline
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --master_port=20001 \
    urbangpt/train/train_mem.py \
    --model_name_or_path ${model_path} \
    --version v1 \
    --data_path ${instruct_ds} \
    --st_content ./TAXI.json \
    --st_data_path ${st_data_path} \
    --st_tower ${pretra_ste} \
    --tune_st_mlp_adapter True \
    --st_select_layer -2 \
    --use_st_start_end \
    --bf16 True \
    --output_dir ${output_model} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb

```

<span id='Evaluating UrbanGPT'/>

## 4. Evaluating UrbanGPT <a href='#all_catelogue'>[Back to Top]</a>

<span id='Preparing Checkpoints and Data'/>

#### 4.1. Preparing Checkpoints and Data <a href='#all_catelogue'>[Back to Top]</a>

* **Checkpoints:** You could try to evaluate UrbanGPT by using your own model or our released checkpoints.
* **Data:** We split test sets for NYC-taxi datasets and make the instruction data for evaluation. Please refer to the [evaluating](https://huggingface.co/datasets/bjdwh/ST_data_urbangpt).

<span id='Running Evaluation'/>

#### 4.2. Running Evaluation <a href='#all_catelogue'>[Back to Top]</a>

You could start the second stage tuning by filling blanks at [urbangpt_eval.sh](urbangpt_eval.sh). There is an example as below: 
```
# to fill in the following path to evaluation!
output_model=./checkpoints/tw2t_multi_reg-cla-gird
datapath=./ST_data_urbangpt/NYC_taxi_cross-region/NYC_taxi.json
st_data_path=./ST_data_urbangpt/NYC_taxi_cross-region/NYC_taxi_pkl.pkl
res_path=./result_test/cross-region/NYC_taxi
start_id=0
end_id=51920
num_gpus=8

python ./urbangpt/eval/run_urbangpt.py --model-name ${output_model}  --prompting_file ${datapath} --st_data_path ${st_data_path} --output_res_path ${res_path} --start_id ${start_id} --end_id ${end_id} --num_gpus ${num_gpus}
```
---------

<!--
## Contact
For any questions or feedback, feel free to contact [Zhonghang Li](mailto:bjdwh.zzh@gmail.com).
-->

## Citation

If you find UrbanGPT useful in your research or applications, please kindly cite:

```
@misc{li2024urbangpt,
      title={UrbanGPT: Spatio-Temporal Large Language Models}, 
      author={Zhonghang Li and Lianghao Xia and Jiabin Tang and Yong Xu and Lei Shi and Long Xia and Dawei Yin and Chao Huang},
      year={2024},
      eprint={2403.00813},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



## Acknowledgements
You may refer to related work that serves as foundations for our framework and code repository, 
[Vicuna](https://github.com/lm-sys/FastChat). We also partially draw inspirations from [GraphGPT](https://github.com/HKUDS/GraphGPT). The design of our website and README.md was inspired by [NExT-GPT](https://next-gpt.github.io/), and the design of our system deployment was inspired by [gradio](https://www.gradio.app) and [Baize](https://huggingface.co/spaces/project-baize/chat-with-baize). Thanks for their wonderful works.
