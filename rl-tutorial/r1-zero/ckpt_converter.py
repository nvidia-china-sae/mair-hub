#!/usr/bin/env python
# encoding: utf-8
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import fire
from glob import glob
from collections import defaultdict


def main(fsdp_checkpoint_path, huggingface_model_path, output_path):
    state_dict = defaultdict(list)

    world_size = 8
    for rank in range(world_size):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        print('loading', filepath)
        this_state_dict = torch.load(filepath)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())

    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    config = AutoConfig.from_pretrained(huggingface_model_path)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)

    #for filepath in glob(f'{fsdp_checkpoint_path}/model_*.pt'):
    #    part_state_dict = torch.load(filepath)
    #    model.load_state_dict(part_state_dict)

    model.save_pretrained(output_path, max_shard_size="10GB")

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    fire.Fire(main)

# python3 ckpt_converter.py qwen2_5_7b_grpo_cl/global_step_480/actor qwen2_5_7b_grpo_cl/global_step_480/actor/huggingface hf/qwen2_5_7b_grpo_cl_step480

# python3 ckpt_converter.py qwen2_5_3b_grpo_orz_cl/global_step_270/actor qwen2_5_3b_grpo_orz_cl/global_step_270/actor/huggingface hf/qwen2_5_3b_grpo_orz_cl_step270

# python3 ckpt_converter.py qwen2_5_3b_grpo_cl3/global_step_133/actor qwen2_5_3b_grpo_cl3/global_step_133/actor/huggingface hf/qwen2_5_3b_grpo_cl3_step133

# python3 ckpt_converter.py qwen2_5_7b_grpo_orz_cl2/global_step_280/actor qwen2_5_7b_grpo_orz_cl2/global_step_280/actor/huggingface hf/qwen2_5_7b_grpo_orz_cl2_step280

# python3 ckpt_converter.py qwen2_5_7b_grpo_nod_random/global_step_180/actor qwen2_5_7b_grpo_nod_random/global_step_180/actor/huggingface hf/qwen2_5_7b_grpo_nod_random_step180

# python3 ckpt_converter.py DAPO-Qwen2.5-7B-Test/global_step_180/actor DAPO-Qwen2.5-7B-Test/global_step_180/actor/huggingface hf/DAPO-Qwen2.5-7B-Test_step180

# python3 ckpt_converter.py DAPO-Qwen2.5-7B-orz-random/global_step_200/actor DAPO-Qwen2.5-7B-orz-random/global_step_200/actor/huggingface hf/DAPO-Qwen2.5-7B-orz-random_step200

"""
## Upload Trained Model

```python
from huggingface_hub import upload_folder

upload_folder(
    folder_path=".",
    repo_id="pe-nlp/Qwen2.5-7b-grpo-orz-cl2-step280",
    repo_type="model",
    token="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",
    ignore_patterns="**/logs/*.txt",
)
```
"""