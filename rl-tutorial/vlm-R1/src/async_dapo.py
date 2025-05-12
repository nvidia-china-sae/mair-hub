# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
    """Asynchronously process a single computation task"""
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
        print(f"Error: Computation timeout: {solution_str[:30]}...")
        return {"score": 0.0, "error": "Computation timeout"}
    except Exception as e:
        print(f"Error: Computation error: {solution_str[:30]}..., error: {e}")
        return {"score": 0.0, "error": str(e)}


async def parallel_compute_scores(evaluation_func, items, executor, max_resp_len=None, overlong_buffer_cfg=None):
    """Process multiple computation tasks in parallel"""
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
    
    # Execute all tasks simultaneously
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        print(f"Error: Parallel computation failed: {e}")
        # If error occurs, try to terminate all processes
        for pid, proc in executor._processes.items():
            try:
                proc.kill()
            except Exception as kill_err:
                print(f"Error: Unable to terminate process: {kill_err}")
        raise
    
    # Process results
    processed_results = []
    for i, (result, item) in enumerate(zip(results, items)):
        if isinstance(result, Exception) or result is None:
            reward = 0.0
            extra_info_dict = {"score": reward, "error": str(result) if result else "Empty result"}
        else:
            # Handle returned result
            if isinstance(result, dict):
                reward = result.get("score", 0.0)
                extra_info_dict = result
            else:
                reward = float(result)
                extra_info_dict = {"score": reward}
        
        # Handle overlong penalty
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
    """Reward manager using true asynchronous processing."""

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
            raise ValueError("base_url must be provided")
        print(f"AsyncDAPO initialized, base_url: {self.base_url}, max_workers: {self.max_workers}")
        
        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"If {overlong_buffer_cfg=} is set, max_resp_len must be provided, but got None"
            )

    def _prepare_items(self, data):
        """Prepare items to be processed"""
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

            # Decode
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
        """Extend this function step by step according to the available dataset"""
        total_start_time = time.time()

        # If rm_scores exists, return directly
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}
        
        # Prepare items
        items = self._prepare_items(data)
        
        # Asynchronously process all samples
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
                
                # Process results
                for result in results:
                    i = result["i"]
                    valid_response_length = result["valid_response_length"]
                    reward = result["reward"]
                    extra_info = result["extra_info"]
                    data_source = result["data_source"]
                    
                    # Update reward_tensor
                    reward_tensor[i, valid_response_length - 1] = reward
                    
                    # Update extra_info
                    for key, value in extra_info.items():
                        reward_extra_info[key].append(value)
                    
                    # Print logic
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
                print(f"Error occurred while processing samples: {e}")
                # If error occurs, ensure we still return a reasonable result
        
        total_time = time.time() - total_start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor