# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
import asyncio
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


async def single_compute_score(evaluation_func, data_source, solution_str, ground_truth, extra_info, executor, timeout=300.0):
    """异步处理单个计算任务"""
    loop = asyncio.get_running_loop()
    try:
        task = asyncio.wait_for(
            loop.run_in_executor(
                executor,
                partial(evaluation_func, data_source, solution_str, ground_truth, extra_info),
            ),
            timeout=timeout,
        )
        return await task
    except asyncio.TimeoutError:
        print(f"Error: 计算超时: {solution_str[:30]}...")
        return {"score": 0.0, "error": "计算超时"}
    except Exception as e:
        print(f"Error: 计算错误: {solution_str[:30]}..., 错误: {e}")
        return {"score": 0.0, "error": str(e)}


async def parallel_compute_scores(evaluation_func, items, executor, max_resp_len=None, overlong_buffer_cfg=None):
    """并行处理多个计算任务"""
    tasks = []
    
    for item in items:
        task = single_compute_score(
            evaluation_func,
            item["data_source"],
            item["response_str"],
            item["ground_truth"],
            item["extra_info"],
            executor,
            timeout=300.0,
        )
        tasks.append(task)
    
    # 同时执行所有任务
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        print(f"Error: 并行计算出错: {e}")
        # 如果出错，尝试终止所有进程
        for pid, proc in executor._processes.items():
            try:
                proc.kill()
            except Exception as kill_err:
                print(f"Error: 无法终止进程: {kill_err}")
        raise
    
    # 处理结果
    processed_results = []
    for i, (result, item) in enumerate(zip(results, items)):
        if isinstance(result, Exception) or result is None:
            reward = 0.0
            extra_info_dict = {"score": reward, "error": str(result) if result else "空结果"}
        else:
            # 处理返回结果
            if isinstance(result, dict):
                reward = result.get("score", 0.0)
                extra_info_dict = result
            else:
                reward = float(result)
                extra_info_dict = {"score": reward}
        
        # 处理超长惩罚
        if overlong_buffer_cfg and overlong_buffer_cfg.get("enable") and max_resp_len is not None:
            overlong_buffer_len = overlong_buffer_cfg.get("len", 0)
            expected_len = max_resp_len - overlong_buffer_len
            exceed_len = item["valid_response_length"] - expected_len
            overlong_penalty_factor = overlong_buffer_cfg.get("penalty_factor", 0.0)
            overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
            reward += overlong_reward
            if overlong_buffer_cfg.get("log", False):
                extra_info_dict["overlong_reward"] = overlong_reward
                extra_info_dict["overlong"] = overlong_reward < 0
        
        processed_results.append({
            "i": item["i"],
            "valid_response_length": item["valid_response_length"],
            "reward": reward,
            "extra_info": extra_info_dict,
            "data_source": item["data_source"],
            "prompt_str": item["prompt_str"],
            "response_str": item["response_str"],
            "ground_truth": item["ground_truth"],
        })
    
    return processed_results


class AsyncDAPORewardManager:
    """使用真正异步处理的奖励管理器。"""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        max_workers=64,
        base_url=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.max_workers = max_workers
        self.base_url = base_url
        
        if self.base_url is None:
            raise ValueError("base_url必须提供")
        print(f"AsyncDAPO初始化，base_url: {self.base_url}, max_workers: {self.max_workers}")
        
        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"如果设置了{overlong_buffer_cfg=}，则必须提供max_resp_len，但获得了None"
            )

    def _prepare_items(self, data):
        """准备待处理的数据项"""
        items = []
        
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # 解码
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {}) or {}
            
            extra_info["url"] = self.base_url
            
            items.append({
                "i": i,
                "valid_response_length": valid_response_length,
                "prompt_str": prompt_str,
                "response_str": response_str,
                "ground_truth": ground_truth,
                "data_source": data_source,
                "extra_info": extra_info,
            })
            
        return items

    def __call__(self, data: DataProto, return_dict: bool = False):
        """根据可用数据集逐步扩展此函数"""
        total_start_time = time.time()

        # 如果有rm_scores，直接返回
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}
        
        # 准备数据项
        items = self._prepare_items(data)
        
        # 异步处理所有样本
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            try:
                results = asyncio.run(
                    parallel_compute_scores(
                        self.compute_score,
                        items,
                        executor,
                        self.max_resp_len,
                        self.overlong_buffer_cfg,
                    )
                )
                
                # 处理结果
                for result in results:
                    i = result["i"]
                    valid_response_length = result["valid_response_length"]
                    reward = result["reward"]
                    extra_info = result["extra_info"]
                    data_source = result["data_source"]
                    
                    # 更新reward_tensor
                    reward_tensor[i, valid_response_length - 1] = reward
                    
                    # 更新extra_info
                    for key, value in extra_info.items():
                        reward_extra_info[key].append(value)
                    
                    # 处理打印逻辑
                    if data_source not in already_print_data_sources:
                        already_print_data_sources[data_source] = 0
                    
                    if already_print_data_sources[data_source] < self.num_examine:
                        already_print_data_sources[data_source] += 1
                        print("[prompt]", result["prompt_str"])
                        print("[response]", result["response_str"])
                        print("[ground_truth]", result["ground_truth"])
                        for key, value in extra_info.items():
                            print(f"[{key}]", value)
                
            except Exception as e:
                print(f"处理样本时发生错误: {e}")
                # 如果出错，确保我们仍然返回一个合理的结果
        
        total_time = time.time() - total_start_time
        print(f"总处理时间: {total_time:.2f}秒")
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor