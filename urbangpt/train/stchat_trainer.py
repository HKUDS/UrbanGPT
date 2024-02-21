import os
import torch
import torch.nn as nn
import sys
from transformers import Trainer
from typing import Dict, Optional, Sequence


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class STChatTrainer(Trainer):

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_st_mlp_adapter', False):
            # sys.stdout = sys.__stdout__
            # Save the model
            _state_dict = state_dict
            if _state_dict is None:
                # Only save the model itself if we are using distributed training
                model_to_save = unwrap_model(self.model)
                _state_dict = model_to_save.state_dict()

            weight_to_save = {}
            keys_to_match = ['st_projector']
            for k, v in _state_dict.items():
                if any(key_match in k for key_match in keys_to_match):
                    weight_to_save[k] = v

            print('output_dir', output_dir)
            current_folder = output_dir.split('/')[-1]
            if "\\" in current_folder:
                current_folder =  current_folder.split('\\')[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "st_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                print('up', mm_projector_folder, current_folder)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                print('down', current_folder)
                torch.save(weight_to_save, os.path.join(output_dir, f'st_projector.bin'))
            # sys.stdout = open(os.devnull, 'w')

        super(STChatTrainer, self)._save(output_dir, state_dict)
