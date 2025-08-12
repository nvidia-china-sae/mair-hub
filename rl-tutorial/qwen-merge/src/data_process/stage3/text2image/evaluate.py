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
 
import json
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm
import argparse
import os
import itertools
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration parameters
OPENAI_API_KEY = "EMPTY"
DEFAULT_MAX_TOKENS = 4000
MAX_WORKERS = 32

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate and filter model response data")
    parser.add_argument("--input_file",
                       type=str,
                       required=True,
                       help="Input JSON file path (data generated from step1)")
    parser.add_argument("--model",
                       type=str,
                       required=True,
                       help="Evaluation model path, e.g.: /path/to/model/Qwen3-32B")
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
    return parser.parse_args()

def extract_content_after_think(response_text):
    """Extract content after </think> tag"""
    think_end_pattern = r'</think>'
    match = re.search(think_end_pattern, response_text, re.IGNORECASE)
    if match:
        content_after_think = response_text[match.end():].strip()
        return content_after_think
    return None

def is_valid_thinking_format(response_text):
    """Check if response follows <think>...</think>... format"""
    has_think_start = re.search(r'<think>', response_text, re.IGNORECASE)
    has_think_end = re.search(r'</think>', response_text, re.IGNORECASE)

    if not (has_think_start and has_think_end):
        return False

    content_after_think = extract_content_after_think(response_text)
    return content_after_think is not None and len(content_after_think.strip()) > 0

def create_evaluation_prompt(question, original_answer, model_response):
    """Create evaluation prompt"""
    prompt = f"""你是一个专业的答案质量评估专家。请评估以下模型回答的质量。

【问题】
{question}

【参考答案】
{original_answer}

【模型回答】
{model_response}

【评估标准】
1. 如果参考答案是确定性的（如数值、选项、固定字段等），模型回答必须与参考答案一致
2. 如果参考答案包含主观性内容，模型回答应与参考答案保持一致，不能有明显矛盾，可以更好但不能相悖
3. 模型回答应该准确、完整、逻辑清晰
4. 模型回答不应包含错误信息或有害内容

请基于以上标准，对模型回答进行评估。 评估结果：优秀 或 不合格 \no_think"""

    return prompt

class ResponseEvaluator:
    def __init__(self, model_name, api_ips, api_base_template, max_concurrent):
        self.model_name = model_name
        self.api_ips = api_ips
        self.api_base_template = api_base_template
        self.ip_cycle = itertools.cycle(api_ips)
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Pre-create client pool for each API server to avoid repeated creation
        self.clients = {}
        self.client_cycle = {}
        for ip in api_ips:
            api_base = api_base_template.format(ip=ip)
            client = AsyncOpenAI(
                api_key=OPENAI_API_KEY,
                base_url=api_base,
                timeout=60.0,
                max_retries=3,
            )
            self.clients[ip] = client

        # Create client cycle
        self.client_cycle = itertools.cycle([(ip, client) for ip, client in self.clients.items()])

        print(f"Pre-created {len(self.clients)} API clients, connection pool ready")

    def get_next_client(self):
        """Get next client (load balancing)"""
        return next(self.client_cycle)

    def create_evaluation_request(self, evaluation_prompt):
        """Create evaluation request"""
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": evaluation_prompt}],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": 0.3,
            "top_p": 0.9,
        }

    async def make_evaluation_request(self, request_params, item_index):
        """Send evaluation request"""
        async with self.semaphore:
            try:
                ip, client = self.get_next_client()
                api_base = self.api_base_template.format(ip=ip)

                completion_response = await client.chat.completions.create(**request_params)
                choice = completion_response.choices[0]

                if choice.finish_reason != "stop":
                    logger.warning(f"Item {item_index} evaluation incomplete (finish_reason: {choice.finish_reason}) ({api_base})")
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
                logger.error(f"Item {item_index} evaluation request failed ({api_base}): {e}")
                return None

    def parse_evaluation_result(self, evaluation_response):
        """Parse evaluation result, extract content after </think>"""
        if not evaluation_response:
            return None

        content_after_think = extract_content_after_think(evaluation_response)
        if not content_after_think:
            logger.warning("Evaluation response format incorrect, missing content after </think>")
            return None

        # Check evaluation result in content after </think>
        has_excellent = "【优秀】" in content_after_think or "优秀" in content_after_think
        has_unqualified = "【不合格】" in content_after_think or "不合格" in content_after_think

        # If both results are present, evaluation is unclear, discard
        if has_excellent and has_unqualified:
            logger.warning(f"Evaluation result unclear, contains both 'excellent' and 'unqualified': {content_after_think[:100]}...")
            return None
        elif has_excellent:
            return "优秀"
        elif has_unqualified:
            return "不合格"
        else:
            logger.warning(f"Cannot parse evaluation result: {content_after_think[:100]}...")
            return None

    async def evaluate_single_item(self, item_with_index):
        """Evaluate single data item"""
        index, item = item_with_index

        try:
            # 1. Check response format
            response_text = item.get("response", "")
            format_valid = is_valid_thinking_format(response_text)

            if not format_valid:
                logger.info(f"Item {index} format invalid, discarded (missing correct <think>...</think> format)")
                return None

            # 2. Extract content after </think>
            model_response = extract_content_after_think(response_text)
            if not model_response:
                logger.info(f"Item {index} cannot extract content after </think>, discarded")
                return None

            # 3. Create evaluation prompt
            question = item.get("query", "")
            original_answer = item.get("original_output", "")

            evaluation_prompt = create_evaluation_prompt(question, original_answer, model_response)

            # 4. Send evaluation request
            request_params = self.create_evaluation_request(evaluation_prompt)
            evaluation_result = await self.make_evaluation_request(request_params, index)

            if not evaluation_result:
                logger.warning(f"Item {index} evaluation failed")
                return None

            # 5. Parse evaluation result
            evaluation_content = evaluation_result["content"]
            assessment = self.parse_evaluation_result(evaluation_content)

            if assessment == "优秀":
                # Keep original data and add evaluation information
                enhanced_item = item.copy()
                enhanced_item.update({
                    "extracted_response": model_response,
                    "evaluation_content": evaluation_content,
                    "assessment": assessment,
                    "evaluation_token_usage": evaluation_result["usage"]
                })
                logger.info(f"Item {index} evaluated as excellent, kept")
                return enhanced_item
            else:
                logger.info(f"Item {index} evaluated as unqualified or parse failed, discarded")
                return None

        except Exception as e:
            logger.error(f"Error evaluating item {index}: {e}")
            return None

    async def evaluate_all_data(self, input_data, output_file):
        """Evaluate all data"""
        print(f"Using model: {self.model_name}")
        print(f"API servers: {self.api_ips}")
        print(f"Output file: {output_file}")
        print(f"Starting evaluation of {len(input_data)} items with {self.semaphore._value} concurrent workers...")
        print("Only keeping data with correct format and evaluated as 'excellent'")

        # Create indexed data list
        indexed_data = [(i, item) for i, item in enumerate(input_data)]

        # Create all evaluation tasks
        tasks = [self.evaluate_single_item(item_with_index) for item_with_index in indexed_data]

        # Execute all tasks concurrently
        excellent_results = []
        total_evaluation_tokens = 0
        processed_count = 0

        async for result in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating data"):
            try:
                item_result = await result
                processed_count += 1

                if item_result is not None:
                    excellent_results.append(item_result)
                    total_evaluation_tokens += item_result["evaluation_token_usage"]["total_tokens"]

            except Exception as e:
                logger.error(f"Error processing evaluation task: {e}")

        # Save excellent results
        self.save_results(excellent_results, output_file)

        print(f"\nEvaluation completed!")
        print(f"Input data count: {len(input_data)}")
        print(f"Passed evaluation: {len(excellent_results)}")
        print(f"Pass rate: {len(excellent_results)/len(input_data):.2%}")
        print(f"Total evaluation tokens used: {total_evaluation_tokens:,}")
        print(f"Results saved to {output_file}")

        return excellent_results

    def save_results(self, results, output_file):
        """Save results to file"""
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit, ensure clients are properly closed"""
        await self.close_clients()

    async def close_clients(self):
        """Close all client connections"""
        for ip, client in self.clients.items():
            try:
                await client.close()
                print(f"Closed client connection: {ip}")
            except Exception as e:
                logger.warning(f"Error closing client {ip}: {e}")

def generate_output_filename(input_file, output_dir):
    """Generate output filename"""
    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    filename = f"evaluated_{input_basename}.json"
    return os.path.join(output_dir, filename)

def load_input_data(input_file):
    """Load input data"""
    print(f"Loading input file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loading completed, {len(data)} items")
    return data

async def main():
    """Main function"""
    args = parse_args()

    # Suggest adjusting excessively high concurrency
    if args.max_workers > 2048:
        logger.warning(f"Concurrency {args.max_workers} might be too high, suggest setting between 512-2048")
        logger.warning("Excessively high concurrency may cause system resource exhaustion or API server connection rejection")

    # Generate output filename
    output_file = generate_output_filename(args.input_file, args.output_dir)

    print(f"Configuration:")
    print(f"  Input file: {args.input_file}")
    print(f"  Evaluation model: {args.model}")
    print(f"  API servers: {args.api_ips}")
    print(f"  API template: {args.api_base_template}")
    print(f"  Max workers: {args.max_workers}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Output file: {output_file}")
    print("-" * 50)

    # Load input data
    input_data = load_input_data(args.input_file)

    # Use async context manager to ensure clients are properly closed
    async with ResponseEvaluator(
        model_name=args.model,
        api_ips=args.api_ips,
        api_base_template=args.api_base_template,
        max_concurrent=args.max_workers
    ) as evaluator:

        results = await evaluator.evaluate_all_data(input_data, output_file)

        # Final statistics
        thinking_excellent = sum(1 for r in results if r.get('mode') == 'thinking')
        no_thinking_excellent = sum(1 for r in results if r.get('mode') == 'no_thinking')

        print(f"\nFinal statistics:")
        print(f"  Excellent thinking responses: {thinking_excellent}")
        print(f"  Excellent no_thinking responses: {no_thinking_excellent}")
        print(f"  Total excellent responses: {len(results)}")

if __name__ == "__main__":
    asyncio.run(main())


# Usage examples:
# # Regular evaluation
# python filter.py --input_file generated_responses_Qwen3-32B.json --model Qwen/Qwen3-32B

# # Using multiple IPs for load balancing
# python filter.py --input_file generated_responses_Qwen3-32B.json --model Qwen/Qwen3-32B --api_ips ip1 ip2  --max_workers 32