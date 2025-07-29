# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from datasets import load_dataset
import json
import random
import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
import argparse
import os
import itertools
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration parameters
OPENAI_API_KEY = "EMPTY"
DEFAULT_MAX_TOKENS = 8000
NO_THINKING_PROBABILITY = 0.5
MAX_WORKERS = 32

# Global variables
all_querys = []
all_original_outputs = []

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate model responses concurrently")
    parser.add_argument("--model",
                       type=str,
                       required=True,
                       help="Model path, e.g.: Qwen/Qwen3-32B")
    parser.add_argument("--api_ips",
                       nargs='+',
                       default=["localhost"],
                       help="API server IP list for load balancing")
    parser.add_argument("--api_base_template",
                       default="http://{ip}:8000/v1",
                       help="API base URL template, {ip} will be replaced with actual IP")
    parser.add_argument("--max_workers",
                       type=int,
                       default=MAX_WORKERS,
                       help="Number of concurrent workers")
    parser.add_argument("--output_dir",
                       type=str,
                       default="./",
                       help="Output directory")
    parser.add_argument("--test_mode",
                       action='store_true',
                       help="Enable test mode with 100 samples")
    return parser.parse_args()

def get_model_name_from_path(model_path):
    """Extract model name from model path"""
    return os.path.basename(model_path)

def generate_output_filename(model_path, output_dir):
    """Generate output filename based on model name"""
    model_name = get_model_name_from_path(model_path)
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    filename = f"generated_responses_{safe_model_name}.json"
    return os.path.join(output_dir, filename)

def load_datasets(test_mode=False):
    """Load datasets"""
    global all_querys, all_original_outputs

    # Load DeepSeek dataset
    print("Loading DeepSeek dataset...")
    ds = load_dataset("Congliu/Chinese-DeepSeek-R1-Distill-data-110k")
    deepseek_queries = []
    deepseek_outputs = []
    for item in ds['train']:
        deepseek_queries.append(item['input'])
        deepseek_outputs.append(item.get('output', ''))

    # Load Alpaca dataset
    print("Loading Alpaca dataset...")
    ds_apache = load_dataset("tatsu-lab/alpaca")['train']
    alpaca_queries = []
    alpaca_outputs = []
    for item in ds_apache:
        assert item['instruction']
        if item['input']:
            alpaca_queries.append(item['instruction'] + "\n" + item['input'])
        else:
            alpaca_queries.append(item['instruction'])
        alpaca_outputs.append(item.get('output', ''))

    print(f"DeepSeek dataset loaded: {len(deepseek_queries)} queries")
    print(f"Alpaca dataset loaded: {len(alpaca_queries)} queries")

    # Apply test mode limit if specified - ensure both datasets are represented
    if test_mode:
        test_samples = 100  # Fixed number when test mode is enabled
        # Split test_samples between datasets proportionally, with minimum 1 sample each
        deepseek_limit = max(1, test_samples // 2)
        alpaca_limit = max(1, test_samples - deepseek_limit)
        
        # Ensure we don't exceed available data
        deepseek_limit = min(deepseek_limit, len(deepseek_queries))
        alpaca_limit = min(alpaca_limit, len(alpaca_queries))
        
        print(f"Test mode enabled: Using {deepseek_limit} DeepSeek + {alpaca_limit} Alpaca samples (total: {deepseek_limit + alpaca_limit})")
        
        deepseek_queries = deepseek_queries[:deepseek_limit]
        deepseek_outputs = deepseek_outputs[:deepseek_limit]
        alpaca_queries = alpaca_queries[:alpaca_limit]
        alpaca_outputs = alpaca_outputs[:alpaca_limit]

    # Combine datasets
    all_querys.extend(deepseek_queries)
    all_querys.extend(alpaca_queries)
    all_original_outputs.extend(deepseek_outputs)
    all_original_outputs.extend(alpaca_outputs)

    total_queries = len(all_querys)
    print(f"Total queries for processing: {total_queries}")

class ModelProcessor:
    def __init__(self, model_name, api_ips, api_base_template, max_concurrent):
        self.model_name = model_name
        self.api_ips = api_ips
        self.api_base_template = api_base_template
        self.ip_cycle = itertools.cycle(api_ips)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.all_successful_data = []

    def get_next_api_base(self) -> str:
        """Get next API address"""
        ip = next(self.ip_cycle)
        return self.api_base_template.format(ip=ip)

    def should_generate_no_thinking(self):
        """Determine whether to generate no_thinking mode response"""
        return random.random() < NO_THINKING_PROBABILITY

    def create_thinking_request(self, query):
        """Create thinking mode request"""
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": query}],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": 0.6,
            "top_p": 0.95,
        }

    def create_no_thinking_request(self, query):
        """Create no_thinking mode request"""
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": query + " /no_think"}],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": 0.7,
            "top_p": 0.8,
        }

    async def make_api_request(self, request_params, mode, query_index):
        """Send async API request and return complete response information"""
        async with self.semaphore:
            try:
                api_base = self.get_next_api_base()
                client = AsyncOpenAI(
                    api_key=OPENAI_API_KEY,
                    base_url=api_base,
                    timeout=60.0
                )

                completion_response = await client.chat.completions.create(**request_params)
                choice = completion_response.choices[0]

                if choice.finish_reason != "stop":
                    logger.warning(f"Query {query_index} {mode} mode incomplete generation (finish_reason: {choice.finish_reason}) ({api_base})")
                    return None

                result_data = {
                    "content": choice.message.content,
                    "finish_reason": choice.finish_reason,
                    "usage": {
                        "prompt_tokens": completion_response.usage.prompt_tokens,
                        "completion_tokens": completion_response.usage.completion_tokens,
                        "total_tokens": completion_response.usage.total_tokens
                    }
                }

                return result_data

            except Exception as e:
                logger.error(f"Query {query_index} {mode} mode request failed ({api_base}): {e}")
                return None

    async def process_single_query(self, query_info):
        """Process single query, may generate 1 or 2 responses"""
        index, query = query_info
        random.seed(42 + index)
        results = []

        try:
            # Generate thinking mode response
            thinking_params = self.create_thinking_request(query)
            thinking_result = await self.make_api_request(thinking_params, "thinking", index)

            if thinking_result is not None:
                results.append({
                    "query_index": index,
                    "query": query,
                    "original_output": all_original_outputs[index],
                    "response": thinking_result["content"],
                    "mode": "thinking",
                    "model": self.model_name,
                    "finish_reason": thinking_result["finish_reason"],
                    "token_usage": thinking_result["usage"]
                })
            else:
                logger.warning(f"Discarded thinking response for query {index} (incomplete generation or request failed)")

            # Generate no_thinking mode response with probability
            if self.should_generate_no_thinking():
                no_thinking_params = self.create_no_thinking_request(query)
                no_thinking_result = await self.make_api_request(no_thinking_params, "no_thinking", index)

                if no_thinking_result is not None:
                    results.append({
                        "query_index": index,
                        "query": query,
                        "original_output": all_original_outputs[index],
                        "response": no_thinking_result["content"],
                        "mode": "no_thinking",
                        "model": self.model_name,
                        "finish_reason": no_thinking_result["finish_reason"],
                        "token_usage": no_thinking_result["usage"]
                    })
                else:
                    logger.warning(f"Discarded no_thinking response for query {index} (incomplete generation or request failed)")

        except Exception as e:
            logger.error(f"Error processing query {index}: {e}")

        return results

    async def process_all_queries(self, output_file):
        """Process all queries asynchronously"""
        query_infos = [(i, query) for i, query in enumerate(all_querys)]

        print(f"Using model: {self.model_name}")
        print(f"API servers: {self.api_ips}")
        print(f"Output file: {output_file}")
        print(f"Starting concurrent processing of {len(all_querys)} queries with {self.semaphore._value} workers...")
        print("Each query generates 100% thinking responses, 50% probability for additional no_thinking responses")
        print("Only keeping complete responses (finish_reason='stop')")

        tasks = [self.process_single_query(query_info) for query_info in query_infos]
        all_results = []
        total_tokens_used = 0

        async for result in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing queries"):
            try:
                query_results = await result
                all_results.extend(query_results)

                for result_item in query_results:
                    total_tokens_used += result_item["token_usage"]["total_tokens"]

            except Exception as e:
                logger.error(f"Error processing task: {e}")

        self.save_results(all_results, output_file)

        print(f"Processing completed!")
        print(f"Generated responses: {len(all_results)}")
        print(f"Total tokens used: {total_tokens_used:,}")
        print(f"Results saved to {output_file}")

        return all_results

    def save_results(self, results, output_file):
        """Save results to file"""
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

async def main():
    """Main function"""
    args = parse_args()

    output_file = generate_output_filename(args.model, args.output_dir)

    test_mode_info = " (Test mode: 100 samples)" if args.test_mode else ""
    print(f"Configuration{test_mode_info}:")
    print(f"  Model path: {args.model}")
    print(f"  API servers: {args.api_ips}")
    print(f"  API template: {args.api_base_template}")
    print(f"  Max workers: {args.max_workers}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Output file: {output_file}")
    if args.test_mode:
        print(f"  Test mode: 100 samples")
    print("-" * 50)

    load_datasets(args.test_mode)

    processor = ModelProcessor(
        model_name=args.model,
        api_ips=args.api_ips,
        api_base_template=args.api_base_template,
        max_concurrent=args.max_workers
    )

    results = await processor.process_all_queries(output_file)

    thinking_count = sum(1 for r in results if r['mode'] == 'thinking')
    no_thinking_count = sum(1 for r in results if r['mode'] == 'no_thinking')

    total_prompt_tokens = sum(r['token_usage']['prompt_tokens'] for r in results)
    total_completion_tokens = sum(r['token_usage']['completion_tokens'] for r in results)
    total_tokens = sum(r['token_usage']['total_tokens'] for r in results)

    print(f"Mode statistics:")
    print(f"  Thinking responses: {thinking_count}")
    print(f"  No-thinking responses: {no_thinking_count}")
    print(f"  No-thinking generation rate: {no_thinking_count/len(all_querys):.2%}")
    print(f"  Average responses per query: {len(results)/len(all_querys):.2f}")

    print(f"Token usage statistics:")
    print(f"  Total input tokens: {total_prompt_tokens:,}")
    print(f"  Total output tokens: {total_completion_tokens:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average tokens per response: {total_tokens/len(results):.0f}")

if __name__ == "__main__":
    asyncio.run(main())
