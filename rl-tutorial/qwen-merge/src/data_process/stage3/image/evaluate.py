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

import os
import random
import asyncio
import aiohttp
import base64
import re
from pathlib import Path
import argparse
from typing import List, Dict, Any
import logging
from tqdm import tqdm
import itertools
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GLMResponseQualityChecker:
    def __init__(self,
                 input_file: str = "glm_response.parquet",
                 output_file: str = "glm_final.parquet",
                 image_base_dir: str = "llavaonevision_converted",
                 api_ips: List[str] = None,
                 api_base_template: str = "http://{ip}:8000/v1",
                 api_key: str = "EMPTY",
                 judge_model_name: str = "THUDM/GLM-4.1V-9B-Thinking",
                 max_workers: int = 16):

        self.input_file = input_file
        self.output_file = output_file
        self.image_base_dir = Path(image_base_dir)

        # Handle IP list
        if api_ips is None:
            api_ips = ["localhost"]
        self.api_ips = api_ips
        self.api_base_template = api_base_template
        self.ip_cycle = itertools.cycle(api_ips)

        self.api_key = api_key
        self.judge_model_name = judge_model_name
        self.max_workers = max_workers

        # Create semaphore for async sessions
        self.semaphore = asyncio.Semaphore(max_workers)

        # Store filtered high-quality data
        self.high_quality_data = []

        # Statistics
        self.stats = {
            "total": 0,
            "format_invalid": 0,
            "box_format_error": 0,
            "multi_turn_discarded": 0,
            "format_valid": 0,
            "judge_success": 0,
            "judge_error": 0,
            "high_quality": 0,
            "low_quality": 0,
            "high_quality_with_box": 0,
            "high_quality_without_box": 0
        }

    def get_next_api_base(self) -> str:
        """Get next API address"""
        ip = next(self.ip_cycle)
        return self.api_base_template.format(ip=ip)

    def check_response_format(self, glm_response: str) -> bool:
        """Check if GLM response format is valid"""
        has_think = "<think>" in glm_response and "</think>" in glm_response
        has_answer = "<answer>" in glm_response

        return has_think and has_answer

    def check_box_format(self, content: str) -> bool:
        """Check if box tags appear in pairs"""
        begin_count = content.count("<|begin_of_box|>")
        end_count = content.count("<|end_of_box|>")

        # Box tags must appear in pairs
        if begin_count != end_count:
            logger.warning(f"Box tags not paired: begin={begin_count}, end={end_count}")
            return False

        return True

    def extract_answer_content(self, glm_response: str) -> str:
        """Extract GLM complete response, remove answer tags and box tags, keep all other content"""
        try:
            # Check box format correctness (for validation, but doesn't affect processing)
            if not self.check_box_format(glm_response):
                return ""

            # Remove answer tags but keep content inside tags
            processed_content = glm_response

            # Remove <answer> start tag
            processed_content = re.sub(r'<answer>', '', processed_content)

            # Remove </answer> end tag (if exists)
            processed_content = re.sub(r'</answer>', '', processed_content)

            # Remove begin_of_box and end_of_box tags but keep content
            processed_content = re.sub(r'<\|begin_of_box\|>', '', processed_content)
            processed_content = re.sub(r'<\|end_of_box\|>', '', processed_content)

            return processed_content.strip()

        except Exception as e:
            logger.error(f"Error processing GLM response content: {e}")
            return ""

    def has_box_tags(self, glm_response: str) -> bool:
        """Check if response contains begin_of_box tags"""
        return "<|begin_of_box|>" in glm_response

    def extract_original_content(self, item: Dict[str, Any]) -> Dict[str, str]:
        """Extract user and assistant content from original data, discard multi-turn conversations"""
        messages = item.get("messages", [])

        # Check if it's multi-turn conversation, discard if so
        if len(messages) > 2:
            logger.debug(f"Discarding multi-turn conversation: {len(messages)} messages, source: {item.get('source_file', 'unknown')}")
            return {"user_content": "", "assistant_content": ""}

        user_content = ""
        assistant_content = ""

        for msg in messages:
            if msg["role"] == "user":
                # Remove various image tags (consistent with process.py)
                content = msg["content"]
                content = content.replace("<image>\n", "").replace("<image>", "")
                content = content.replace("<|image|>\n", "").replace("<|image|>", "")
                user_content = content.strip()

            elif msg["role"] == "assistant":
                assistant_content = msg["content"].strip()

        return {
            "user_content": user_content,
            "assistant_content": assistant_content
        }

    def prepare_image(self, image_path: str) -> str:
        """Prepare image data, convert to base64 format"""
        try:
            full_image_path = self.image_base_dir / image_path
            if not full_image_path.exists():
                logger.warning(f"Image file not found: {full_image_path}")
                return None

            with open(full_image_path, "rb") as f:
                encoded_image = base64.b64encode(f.read())
            encoded_image_text = encoded_image.decode("utf-8")
            return f"data:image;base64,{encoded_image_text}"
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None

    def create_quality_check_prompt(self, user_content: str, original_answer: str, glm_answer: str, has_box: bool) -> str:
        """Create quality check prompt"""

        base_instruction = """
你是一个专业的教育内容质量评估专家。你的任务是评估GLM模型生成的回答质量。

**背景说明**:
我们正在升级教育数据集的质量。原始数据包含问题和答案，我们用更先进的GLM模型重新回答了这些问题，现在需要你判断GLM的回答质量。

**评估标准**:
1. **正确性**: GLM答案必须正确回答问题，没有事实性错误
2. **完整性**: 回答需要全面，涵盖问题的所有要点
3. **清晰性**: 解释清楚，逻辑连贯，易于理解
4. **专业性**: 使用恰当的专业术语和表达方式
"""

        if has_box:
            specific_instruction = """
**特别注意**: GLM答案中包含了特殊标记框，这表示问题有明确的标准答案（如数学计算题、选择题等）。

**评估策略**:
1. **优先级1**: 如果GLM答案与原始答案一致且都正确，直接判定为"优秀"
2. **优先级2**: 如果GLM答案与原始答案不一致， 则判定为"不合格"
3. **优先级3**: 如果无法明确判断正确性，参考原始答案的合理性，如果GLM答案明显优于原始答案，可以判定为"优秀"

严格要求：答案必须正确，解题步骤完整清晰。
"""
        else:
            specific_instruction = """
**特别注意**: 这是一个开放性问题，没有唯一标准答案。

**评估策略**:
1. **优先级1**: 如果GLM答案与原始答案在核心观点上一致且都合理，判定为"优秀"
2. **优先级2**: 如果GLM答案与原始答案不完全一致，评估GLM答案的质量：
   - 是否比原始答案更详细、更准确
   - 是否提供了更好的解释和分析
   - 是否展现了更深入的理解
3. **优先级3**: 参考原始答案作为基准，如果GLM答案质量明显不如原始答案，判定为"不合格"

综合考虑：整体质量是否达到或超过原始答案水平。
"""

        prompt = f"""{base_instruction}

{specific_instruction}

**问题内容**:
{user_content}

**原始答案**:
{original_answer}

**GLM重新生成的答案**:
{glm_answer}

**评估任务**:
请按照上述评估策略，仔细分析GLM答案的质量。优先判断答案的一致性和正确性，在无法明确判断时参考原始答案进行评估。

请将你的最终判断结果放在<|begin_of_box|>和<|end_of_box|>标签中，只回答"优秀"或"不合格"。

同时请简要说明你的判断理由，包括GLM答案的优点和可能存在的问题。"""

        return prompt

    def extract_quality_result(self, judge_response: str) -> str:
        """Extract excellent/unqualified from judgment result"""
        try:
            # First try to extract from box tags
            box_match = re.search(r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>', judge_response, re.DOTALL)
            if box_match:
                content = box_match.group(1).strip()
                if "优秀" in content or "excellent" in content.lower():
                    return "优秀"
                elif "不合格" in content or "unqualified" in content.lower() or "不符合" in content:
                    return "不合格"

            # If no box tags, extract directly from response
            if "优秀" in judge_response or "excellent" in judge_response.lower():
                return "优秀"
            elif "不合格" in judge_response or "unqualified" in judge_response.lower() or "不符合" in judge_response:
                return "不合格"

            logger.warning(f"Unable to parse quality judgment result: {judge_response[:200]}...")
            return "错误"

        except Exception as e:
            logger.error(f"Error parsing quality judgment result: {e}")
            return "错误"

    def prepare_judge_messages(self, user_content: str, original_answer: str,
                             glm_answer: str, has_box: bool, image_base64: str = None) -> List[Dict[str, Any]]:
        """Prepare judgment message format, including image"""
        judge_prompt = self.create_quality_check_prompt(user_content, original_answer, glm_answer, has_box)

        if image_base64:
            # Message format with image
            message_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_base64
                    },
                },
                {"type": "text", "text": judge_prompt}
            ]
            return [{"role": "user", "content": message_content}]
        else:
            # Text-only message
            return [{"role": "user", "content": judge_prompt}]

    async def check_quality(self, session: aiohttp.ClientSession,
                          user_content: str, original_answer: str,
                          glm_answer: str, has_box: bool, image_path: str = None) -> str:
        """Async request to judgment model to check quality"""
        async with self.semaphore:
            try:
                # Get next API address
                api_base = self.get_next_api_base()

                # Prepare image
                image_base64 = None
                if image_path:
                    image_base64 = self.prepare_image(image_path)

                # Prepare messages
                messages = self.prepare_judge_messages(
                    user_content, original_answer, glm_answer, has_box, image_base64
                )

                payload = {
                    "model": self.judge_model_name,
                    "messages": messages,
                }

                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                async with session.post(
                    f"{api_base}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        judge_response = result["choices"][0]["message"]["content"].strip()

                        quality_result = self.extract_quality_result(judge_response)

                        return quality_result
                    else:
                        error_text = await response.text()
                        logger.error(f"Quality check API request failed ({api_base}): {response.status}, {error_text}")
                        return "错误"
            except Exception as e:
                logger.error(f"Error requesting quality check model: {e}")
                return "错误"

    async def process_all_data(self):
        """Process all data"""
        logger.info(f"Starting to read file: {self.input_file}")

        try:
            df = pd.read_parquet(self.input_file)
            data = df.to_dict('records')
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            return

        self.stats["total"] = len(data)
        logger.info(f"Total loaded {len(data)} data items")

        # Step 1: Format check and content extraction
        valid_items = []
        empty_content_count = 0

        for item in tqdm(data, desc="Checking format"):
            # First check if it's multi-turn conversation
            messages = item.get("messages", [])
            if len(messages) > 2:
                self.stats["multi_turn_discarded"] += 1
                logger.debug(f"Discarding multi-turn conversation: {len(messages)} messages, source: {item.get('source_file', 'unknown')}")
                continue

            glm_response = item.get("glm4v_response", "")

            if not self.check_response_format(glm_response):
                self.stats["format_invalid"] += 1
                continue

            # Extract content
            glm_answer = self.extract_answer_content(glm_response)
            if not glm_answer:
                self.stats["box_format_error"] += 1
                continue

            self.stats["format_valid"] += 1

            original_content = self.extract_original_content(item)

            # Check if user content is valid
            if not original_content["user_content"]:
                empty_content_count += 1
                self.stats["format_invalid"] += 1
                continue

            has_box = self.has_box_tags(glm_response)

            # Get image path
            image_path = None
            if item.get("images") and len(item["images"]) > 0:
                image_path = item["images"][0]

            valid_item = {
                "item": item,
                "glm_answer": glm_answer,
                "user_content": original_content["user_content"],
                "assistant_content": original_content["assistant_content"],
                "has_box": has_box,
                "image_path": image_path
            }

            valid_items.append(valid_item)

        logger.info(f"Filtered out {empty_content_count} items with empty user content")
        logger.info(f"Format check completed: valid {self.stats['format_valid']}, invalid {self.stats['format_invalid']}, box format error {self.stats['box_format_error']}")

        if not valid_items:
            logger.warning("No valid format data, skipping quality check")
            return

        # Step 2: Concurrent quality check
        connector = aiohttp.TCPConnector(limit=self.max_workers)
        async with aiohttp.ClientSession(connector=connector) as session:

            # Prepare all quality check tasks
            quality_tasks = []
            for valid_item in valid_items:
                task = self.check_quality(
                    session,
                    valid_item["user_content"],
                    valid_item["assistant_content"],
                    valid_item["glm_answer"],
                    valid_item["has_box"],
                    valid_item["image_path"]
                )
                quality_tasks.append(task)

            logger.info(f"Starting concurrent quality check for {len(quality_tasks)} samples")

            # Execute quality check tasks concurrently
            with tqdm(total=len(quality_tasks), desc="Checking quality", unit="items") as pbar:
                quality_results = []
                completed_tasks = asyncio.as_completed(quality_tasks)

                for completed_task in completed_tasks:
                    try:
                        result = await completed_task
                        quality_results.append(result)
                    except Exception as e:
                        logger.error(f"Quality check task execution error: {e}")
                        quality_results.append("错误")

                    pbar.update(1)

            # Step 3: Filter data based on quality check results
            for valid_item, quality_result in zip(valid_items, quality_results):
                if quality_result == "错误":
                    self.stats["judge_error"] += 1
                    continue

                self.stats["judge_success"] += 1

                if quality_result == "优秀":
                    self.stats["high_quality"] += 1
                    if valid_item["has_box"]:
                        self.stats["high_quality_with_box"] += 1
                    else:
                        self.stats["high_quality_without_box"] += 1

                    # Create upgraded data item, replace original answer with GLM answer
                    upgraded_item = valid_item["item"].copy()

                    # Replace assistant content in messages
                    new_messages = []
                    for msg in upgraded_item.get("messages", []):
                        if msg["role"] == "assistant":
                            new_msg = msg.copy()
                            new_msg["content"] = valid_item["glm_answer"]
                            new_messages.append(new_msg)
                        else:
                            new_messages.append(msg)

                    upgraded_item["messages"] = new_messages

                    # Only keep images and messages fields for final output
                    final_item = {
                        "images": upgraded_item.get("images", []),
                        "messages": new_messages
                    }

                    self.high_quality_data.append(final_item)
                else:
                    self.stats["low_quality"] += 1

    def save_upgraded_results(self):
        """Save upgraded results"""
        if self.high_quality_data:
            output_file = Path(self.output_file)

            with tqdm(total=1, desc="Saving final results", unit="file") as pbar:
                df = pd.DataFrame(self.high_quality_data)
                df.to_parquet(output_file, index=False)
                pbar.update(1)

            logger.info(f"Final high-quality dataset saved to: {output_file}")
            logger.info(f"Final retained: {len(self.high_quality_data)} high-quality data items")
            logger.info(f"Each data item contains only images and messages fields")
        else:
            logger.warning("No high-quality data, skipping save")

        # Print statistics
        logger.info("=" * 60)
        logger.info("Data Quality Upgrade Statistics:")
        logger.info(f"Total data: {self.stats['total']}")
        logger.info(f"Multi-turn discarded: {self.stats['multi_turn_discarded']}")
        logger.info(f"Format valid: {self.stats['format_valid']}")
        logger.info(f"Format invalid: {self.stats['format_invalid']}")
        logger.info(f"Box format error: {self.stats['box_format_error']}")
        logger.info(f"Quality check success: {self.stats['judge_success']}")
        logger.info(f"Quality check error: {self.stats['judge_error']}")
        logger.info(f"High quality data: {self.stats['high_quality']}")
        logger.info(f"  - With box tags (clear answers): {self.stats['high_quality_with_box']}")
        logger.info(f"  - Without box tags (open questions): {self.stats['high_quality_without_box']}")
        logger.info(f"Low quality data: {self.stats['low_quality']}")
        if self.stats['total'] > 0:
            success_rate = self.stats['high_quality']/self.stats['total']*100
            logger.info(f"Data upgrade success rate: {self.stats['high_quality']}/{self.stats['total']} = {success_rate:.2f}%")
        logger.info("=" * 60)

    async def run(self):
        """Run the entire processing pipeline"""
        await self.process_all_data()
        self.save_upgraded_results()

def main():
    parser = argparse.ArgumentParser(description="GLM data quality upgrade: replace original answers with high-quality GLM answers")
    parser.add_argument("--input_file", default="glm_response.parquet",
                       help="GLM response result file")
    parser.add_argument("--output_file", default="glm_final.parquet",
                       help="Output file for high-quality data")
    parser.add_argument("--image_base_dir", default="llavaonevision_converted",
                       help="Image base directory")
    parser.add_argument("--api_ips", nargs='+', default=["ptyche0055", "ptyche0057", "ptyche0058", "ptyche0059", "ptyche0060", "ptyche0061", "ptyche0062", "ptyche0063", "ptyche0334", "ptyche0336", "ptyche0337", "ptyche0338", "ptyche0339", "ptyche0340", "ptyche0341", "ptyche0342"],
                       help="API server IP list for load balancing")
    parser.add_argument("--api_base_template", default="http://{ip}:8000/v1",
                       help="API base URL template, {ip} will be replaced with actual IP")
    parser.add_argument("--api_key", default="EMPTY", help="API key")
    parser.add_argument("--judge_model", default="THUDM/GLM-4.1V-9B-Thinking",
                       help="Model name for quality checking")
    parser.add_argument("--max_workers", type=int, default=1024, help="Maximum number of workers")

    args = parser.parse_args()

    processor = GLMResponseQualityChecker(
        input_file=args.input_file,
        output_file=args.output_file,
        image_base_dir=args.image_base_dir,
        api_ips=args.api_ips,
        api_base_template=args.api_base_template,
        api_key=args.api_key,
        judge_model_name=args.judge_model,
        max_workers=args.max_workers
    )

    # Run async processing
    asyncio.run(processor.run())

if __name__ == "__main__":
    main()