#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
    LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from urbangpt.model.st_layers import ST_Enc, parse_args
import json
import os.path as osp
import glob


IGNORE_INDEX = -100
DEFAULT_STHIS_TOKEN = "<ST_HIS>"
DEFAULT_STPRE_TOKEN = "<ST_PRE>"
DEFAULT_ST_PATCH_TOKEN = "<ST_patch>"
DEFAULT_ST_START_TOKEN = "<ST_start>"
DEFAULT_ST_END_TOKEN = "<ST_end>"

def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    mae_loss = torch.abs(true - pred)
    return torch.mean(mae_loss)

def scaler_mae_loss(scaler=None, mask_value=None):
    def loss(preds, labels, mask=None):
        if scaler is not None:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        if mask is not None:
            preds = preds * mask
            labels = labels * mask
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

class STLlamaConfig(LlamaConfig):
    model_type = "STLlama"

class STPretrainConfig:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


def load_model_pretrained(model_name, pretrain_model_path):
    # load conig json
    print("************************", pretrain_model_path)
    assert osp.exists(osp.join(pretrain_model_path, 'config.json')), 'config.json missing'
    with open(osp.join(pretrain_model_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    args = STPretrainConfig(config_dict)
    model = model_name(args)
    pkl_files = glob.glob(osp.join(pretrain_model_path, '*.pkl'))
    state_dict = torch.load(pkl_files[0])
    # print(state_dict.keys())
    if 'logit_scale' in state_dict.keys():
        state_dict.pop('logit_scale')
    print('loading ST pre train model')
    model.load_state_dict(state_dict)

    return model, args


class STLlamaModel(LlamaModel):
    config_class = STLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(STLlamaModel, self).__init__(config)
        self.st_start_id0 = []
        self.st_start_id1 = []
        self.pre_STE = None

        if hasattr(config, "st_tower"):
            if self.config.st_tower == "ST_Encoder":
                args = parse_args()
                # torch.backends.cudnn.enabled = False
                # torch.cuda.cudnn_enabled = True
                # torch.backends.cudnn.deterministic = False

                self.st_tower = ST_Enc(args, dim_in=2, dim_out=2)
                loaded_state_dict = torch.load(self.config.pretrain_ST_model_path)
                model_state_dict = self.st_tower.state_dict()
                for name, param in loaded_state_dict.items():
                    if 'predictor' in name:
                        new_name = name.replace('predictor.', '')
                        model_state_dict[new_name].copy_(param)

        if hasattr(config, "use_st_proj"):
            self.st_projector = nn.Linear(config.st_hidden_size, config.hidden_size)

    def set_st_tower(self):
        st_tower = getattr(self, 'st_tower', None)
        if type(st_tower) is list:
            st_tower = st_tower[0]

        st_tower = st_tower.to(dtype=torch.float32)
        loaded_state_dict = torch.load(self.config.pretrain_ST_model_path)
        model_state_dict = st_tower.state_dict()
        for name, param in loaded_state_dict.items():
            if 'predictor' in name:
                new_name = name.replace('predictor.', '')
                model_state_dict[new_name].copy_(param)
        return st_tower

    def get_st_tower(self):
        st_tower = getattr(self, 'st_tower', None)
        if type(st_tower) is list:
            st_tower = st_tower[0]
        return st_tower

    def initialize_st_modules(self, st_tower, st_select_layer,
                                 pretrain_st_mlp_adapter=None, fsdp=None):  # TODO: modify this function
        self.config.st_tower = st_tower

        if not hasattr(self, 'st_tower'):
            if self.config.st_tower == "ST_Encoder":
                args = parse_args()
                torch.backends.cudnn.enabled = False
                # torch.cuda.cudnn_enabled = True
                # torch.backends.cudnn.deterministic = False

                st_tower = ST_Enc(args, dim_in=2, dim_out=2)
                loaded_state_dict = torch.load(self.config.pretrain_ST_model_path)
                model_state_dict = st_tower.state_dict()
                for name, param in loaded_state_dict.items():
                        new_name = name.replace('predictor.', '')
                        model_state_dict[new_name].copy_(param)
        else:
            st_tower = self.st_tower

        if fsdp is not None and len(fsdp) > 0:
            self.st_tower = [st_tower]
        else:
            self.st_tower = st_tower

        self.config.use_st_proj = True
        self.config.st_select_layer = st_select_layer

        if not hasattr(self, 'st_projector'):
            self.st_projector = nn.Linear(self.config.st_hidden_size, self.config.hidden_size)

        if pretrain_st_mlp_adapter is not None:
            st_projector_weights = torch.load(pretrain_st_mlp_adapter, map_location='cpu')
            self.st_projector.load_state_dict({k.split('.')[-1]: v for k, v in st_projector_weights.items()})

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            st_data_x: Optional[list] = None,
            st_data_y: Optional[list] = None,
            region_start: Optional[int] = -1,
            region_end: Optional[int] = -1,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if len(st_data_x) > 1:
            st_data_x = torch.cat(st_data_x, dim=0)
            st_data_y = torch.cat(st_data_y, dim=0)

        st_tower = self.get_st_tower()
        if st_tower is not None and (input_ids.shape[1] != 1 or self.training) and st_data_x is not None:
            if type(st_data_x) is list:
                # variable length images
                pre_STE, STE_out = st_tower(st_data_x[0][..., :2])
                _, STE_lbls_out = st_tower(st_data_y[0][..., :2])
                if STE_out.shape[2] > 1:
                    region_select_out = STE_out[:, :, region_start[0]:region_end[0], :].to(torch.bfloat16)
                else:
                    region_select_out = STE_out.to(torch.bfloat16)
            else:
                batch_size = st_data_x.shape[0]
                pre_STE, STE_out = st_tower(st_data_x[..., :2])
                _, STE_lbls_out = st_tower(st_data_y[..., :2])
                region_select_list = []
                for i in range(batch_size):
                    region_select_out = STE_out[i:i+1, :, region_start[i]:region_end[i], :].to(torch.bfloat16)
                    region_select_list.append(region_select_out)
                region_select_out = torch.cat(region_select_list, dim=0)
            self.pre_STE = pre_STE
            st_projector_out = self.st_projector(region_select_out.transpose(1, -1))

            new_input_embeds = []
            cur_st_idx = 0
            self.st_start_id0 = []
            self.st_start_id1 = []
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                # if st_tower.config.use_st_start_end:
                cur_st_features = st_projector_out[cur_st_idx]
                cur_st_features = cur_st_features.reshape(cur_st_features.shape[0], -1)
                num_patches = cur_st_features.shape[0]
                if (cur_input_ids == st_tower.config.st_start_token).sum() != (
                        cur_input_ids == st_tower.config.st_end_token).sum():
                    raise ValueError("The number of st start tokens and st end tokens should be the same.")
                st_start_tokens = torch.where(cur_input_ids == st_tower.config.st_start_token)[0]
                # st_end_tokens = torch.where(cur_input_ids == st_tower.config.st_end_token)[0]

                if st_start_tokens.shape[0] >= 3:
                    st_start_token_pos1 = st_start_tokens[0]
                    st_start_token_pos2 = st_start_tokens[1]
                    st_start_token_pos3 = st_start_tokens[2]
                    self.st_start_id0.append(st_start_token_pos1)
                    self.st_start_id1.append(st_start_token_pos3)

                    if cur_input_ids[
                        st_start_token_pos1 + num_patches + 1] != st_tower.config.st_end_token:
                        raise ValueError("The st end token should follow the st start token.")

                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:st_start_token_pos1].detach(),
                                                          cur_input_embeds[st_start_token_pos1:st_start_token_pos1 + 1],
                                                          cur_st_features,
                                                          cur_input_embeds[st_start_token_pos1 + num_patches + 1:st_start_token_pos1 + num_patches + 2],
                                                          cur_input_embeds[st_start_token_pos1 + num_patches + 2:st_start_token_pos2].detach(),
                                                          cur_input_embeds[st_start_token_pos2:st_start_token_pos2 + num_patches + 2],
                                                          cur_input_embeds[st_start_token_pos2 + num_patches + 2:st_start_token_pos3].detach(),
                                                          cur_input_embeds[st_start_token_pos3:st_start_token_pos3 + num_patches + 2],
                                                          cur_input_embeds[st_start_token_pos3 + num_patches + 2:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:st_start_token_pos1 + 1],
                                                          cur_st_features,
                                                          cur_input_embeds[st_start_token_pos1 + num_patches + 1:]), dim=0)
                    cur_st_idx += 1
                else:
                    st_start_token_pos = st_start_tokens[0]
                    self.st_start_id0.append(st_start_token_pos)
                    num_patches = cur_st_features.shape[0]

                    if cur_input_ids[st_start_token_pos + num_patches + 1] != st_tower.config.st_end_token:
                        raise ValueError("The st end token should follow the st start token.")

                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:st_start_token_pos].detach(),
                                                          cur_input_embeds[st_start_token_pos:st_start_token_pos + 1],
                                                          cur_st_features,
                                                          cur_input_embeds[st_start_token_pos + num_patches + 1:st_start_token_pos + num_patches + 2],
                                                          cur_input_embeds[st_start_token_pos + num_patches + 2:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:st_start_token_pos + 1],
                                                          cur_st_features,
                                                          cur_input_embeds[st_start_token_pos + num_patches + 1:]),dim=0)
                    cur_st_idx += 1
                new_input_embeds.append(cur_new_input_embeds)

            assert cur_st_idx == len(st_projector_out)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(STLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class STLlamaForCausalLM(LlamaForCausalLM):
    config_class = STLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = STLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.st_pred_linear_1 = nn.Linear(self.config.hidden_size, self.config.lin_hidden_size)
        self.st_pred_linear_2 = nn.Linear(self.config.lin_hidden_size*2, self.config.time_steps)
        self.st_pred_linear_3 = nn.Linear(self.config.hidden_size, self.config.lin_hidden_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.st_pre_res = []

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def get_st_tower(self):
        return self.get_model().get_st_tower()

    def set_st_tower(self):
        self.get_model().set_st_tower()

    def get_vision_tower(self):
        model = self.get_model()
        st_tower = model.st_tower
        if type(st_tower) is list:
            st_tower = st_tower[0]
        return st_tower

    def get_st_pre_res(self):
        return self.st_pre_res

    def reset_st_pre_res(self):
        self.st_pre_res = []

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            st_data_x: Optional[list] = None,
            st_data_y: Optional[list] = None,
            region_start: Optional[int] = -1,
            region_end: Optional[int] = -1,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            st_data_x=st_data_x,
            st_data_y=st_data_y,
            region_start=region_start,
            region_end=region_end
        )

        feature_nums = 2
        hidden_states = outputs[0]
        batch_size = hidden_states.shape[0]
        st_pre_embs1_list = []
        st_pre_embs2_list = []

        if labels is not None:
            for bs in range(batch_size):
                st_pre_embs1_batch = hidden_states[bs:bs + 1,
                                     self.model.st_start_id0[bs] + 1:self.model.st_start_id0[bs] + feature_nums + 1,
                                     :].detach()
                st_pre_embs1_list.append(st_pre_embs1_batch)
            st_pre_embs1 = torch.cat(st_pre_embs1_list, dim=0).reshape(batch_size, -1, feature_nums, self.config.hidden_size)
            # # [4, 1, 2, 4096]-->[4, 1, 2, 128]
            st_pre_out1 = self.relu(self.st_pred_linear_1(st_pre_embs1))
            for bs in range(batch_size):
                st_pre_embs2_batch = hidden_states[bs:bs + 1,
                                     self.model.st_start_id1[bs] + 1:self.model.st_start_id1[bs] + feature_nums + 1,
                                     :]
                st_pre_embs2_list.append(st_pre_embs2_batch)
            st_pre_embs2 = torch.cat(st_pre_embs2_list, dim=0).reshape(batch_size, -1, feature_nums, self.config.hidden_size)

            # # [4, 1, 2, 4096]-->[4, 1, 2, 128]
            st_pre_out2 = self.relu(self.st_pred_linear_3(st_pre_embs2))
            # # [4, 1, 2, 256]-->[4, 1, 2, 12]
            st_pre_final = self.st_pred_linear_2(torch.cat([st_pre_out1, st_pre_out2], dim=-1))
            # # [4, 1, 2, 12]-->[4, 1, 12, 2]
            st_pre_final = st_pre_final.transpose(-1, -2)
        else:
            self.st_pre_res.append(hidden_states.clone())

        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            rec_loss = scaler_mae_loss(scaler=None, mask_value=None)
            bce_loss = BCEWithLogitsLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)

            if len(st_data_y) > 1:
                st_data_y = torch.cat(st_data_y, dim=0)
                labels_stpre = st_data_y[:, :, region_start[0]:region_end[0], :feature_nums].transpose(1, 2).to(torch.bfloat16)
                task_type_all = st_data_y[:, 0, region_start[0], -1]
            else:
                labels_stpre = st_data_y[0][0:1, :, region_start[0]:region_end[0], :feature_nums].transpose(1, 2).to(torch.bfloat16)
                task_type_all = st_data_y[0][0:1, 0, region_start[0], -1]

            regress_idx_list = []
            classificate_idx_list = []
            regress_result_list = []
            classificate_result_list = []
            for i in range(batch_size):
                task_type = task_type_all[i]
                # classification
                if task_type == 3 or task_type == 4:
                    classificate_idx_list.append(i)
                    regress_result_list.append(st_pre_final[i:i + 1, ...].detach())
                    classificate_result_list.append(st_pre_final[i:i + 1, ...])
                # regression
                else:
                    regress_idx_list.append(i)
                    classificate_result_list.append(st_pre_final[i:i + 1, ...].detach())
                    regress_result_list.append(st_pre_final[i:i + 1, ...])
            regress_result = torch.cat(regress_result_list, dim=0)
            classificate_result = torch.cat(classificate_result_list, dim=0)


            loss_regress = rec_loss(regress_result, labels_stpre)
            labels_classificate = labels_stpre
            labels_classificate[labels_classificate >= 1] = 1
            labels_classificate[labels_classificate < 1] = 0
            loss_classificate = bce_loss(classificate_result, labels_classificate)

            loss = loss_fct(shift_logits, shift_labels) + loss_regress + loss_classificate

        if not return_dict:
            # print('not return dict')
            output = (logits,) + outputs[1:]
            print(loss.shape)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # print(sss)
        if past_key_values:
            # print('past_key_values', input_ids.shape)
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                # "st_data": [kwargs.get("st_data", None)],
                "st_data_x": [kwargs.get("st_data_x", None)],
                "st_data_y": [kwargs.get("st_data_y", None)],
                "region_start": [kwargs.get("region_start", None)],
                "region_end": [kwargs.get("region_end", None)]
                # "edge_index_reps": kwargs.get("edge_index_reps", None),
            }
        )
        # print('model_inputs.update')
        return model_inputs

    def reset_lm_head(self):
        self.get_input_embeddings().weight.data[-3:, :] = self.lm_head_add.weight.data

    def initialize_st_tokenizer(self, use_st_start_end, tokenizer, device,
                                   tune_st_mlp_adapter=False, pretrain_st_mlp_adapter=None):
        vision_config = self.get_st_tower().config
        vision_config.use_st_start_end = use_st_start_end
        tokenizer.add_tokens([DEFAULT_ST_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if use_st_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_ST_START_TOKEN, DEFAULT_ST_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.st_start_token, vision_config.st_end_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_ST_START_TOKEN, DEFAULT_ST_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_st_mlp_adapter:
                self.get_model().orig_embeds_params = [
                    self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True

            if pretrain_st_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_st_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        vision_config.st_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_ST_PATCH_TOKEN])[0]

AutoConfig.register("STLlama", STLlamaConfig)
AutoModelForCausalLM.register(STLlamaConfig, STLlamaForCausalLM)
