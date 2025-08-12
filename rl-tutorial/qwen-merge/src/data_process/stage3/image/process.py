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
import os
import random
import asyncio
import aiohttp
import base64
from pathlib import Path
import argparse
from typing import List, Dict, Any
import logging
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
import itertools
import pandas as pd
import tempfile
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GLMDataProcessor:
    def __init__(self,
                 data_dir: str = ".",
                 api_ips: List[str] = None,
                 api_base_template: str = "http://{ip}:8000/v1",
                 api_key: str = "EMPTY",
                 model_name: str = "THUDM/GLM-4.1V-9B-Thinking",
                 max_workers: int = 32,
                 samples_per_file: int = 64,
                 test_mode: bool = False):

        self.data_dir = Path(data_dir)

        # Handle IP list
        if api_ips is None:
            api_ips = ["localhost"]
        self.api_ips = api_ips
        self.api_base_template = api_base_template
        self.ip_cycle = itertools.cycle(api_ips)

        self.api_key = api_key
        self.model_name = model_name
        self.max_workers = max_workers
        self.samples_per_file = samples_per_file
        self.test_mode = test_mode

        # Create semaphore for async sessions
        self.semaphore = asyncio.Semaphore(max_workers)

        # Store all successfully processed data
        self.all_successful_data = []

    def get_next_api_base(self) -> str:
        """Get next API address"""
        ip = next(self.ip_cycle)
        return self.api_base_template.format(ip=ip)

    def should_skip_file(self, filename: str) -> bool:
        """Check if file should be skipped"""
        filename_lower = filename.lower()
        skip_keywords = ['cc3m', '558k']

        for keyword in skip_keywords:
            if keyword in filename_lower:
                logger.info(f"Skipping file {filename}: contains keyword '{keyword}'")
                return True
        return False

    def get_parquet_files(self) -> List[Path]:
        """Get all parquet files, filtering out unwanted files"""
        all_parquet_files = list(self.data_dir.glob("*.parquet"))
        parquet_files = [f for f in all_parquet_files if not self.should_skip_file(f.name)]

        skipped_count = len(all_parquet_files) - len(parquet_files)
        logger.info(f"Found {len(all_parquet_files)} parquet files, skipped {skipped_count} files, processing {len(parquet_files)} files")

        return parquet_files

    def validate_item(self, item: Dict[str, Any]) -> bool:
        """Validate if data item is valid"""
        if not isinstance(item, dict):
            return False

        messages = item.get("messages", [])
        # Handle numpy array case
        if hasattr(messages, '__len__'):
            if len(messages) == 0:
                return False
        else:
            return False

        # Convert to list if it's a numpy array
        if hasattr(messages, 'tolist'):
            messages = messages.tolist()

        # Check if there are user messages
        user_messages = [msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "user"]
        if len(user_messages) == 0:
            return False

        # Check first user message content
        first_user_msg = user_messages[0]
        content = first_user_msg.get("content", "")
        
        # Handle numpy string case
        if hasattr(content, 'item'):
            content = content.item()
        content = str(content)

        # Check if there's still text content after removing image tags
        clean_content = content.replace("<image>", "").replace("<|image|>", "").strip()
        if not clean_content:
            return False

        # Check if there are images
        images = item.get("images", [])
        # Handle numpy array case
        if hasattr(images, '__len__'):
            if len(images) == 0:
                return False
        else:
            return False

        return True

    def load_and_sample_all_data(self) -> List[tuple[Dict[str, Any], str]]:
        """Load all parquet files and sample, return list of (data_item, source_filename) tuples"""
        all_sampled_data = []
        parquet_files = self.get_parquet_files()

        mode_info = " (Test mode)" if self.test_mode else ""
        logger.info(f"Starting to load and sample all data{mode_info}...")

        if self.test_mode:
            max_total_test_samples = 100
            samples_per_file_test = max(1, max_total_test_samples // len(parquet_files))
            logger.info(f"Test mode: each file will sample up to {samples_per_file_test} items")

        for parquet_file in tqdm(parquet_files, desc="Loading files", unit="files"):
            try:
                # Load parquet file using pandas
                df = pd.read_parquet(parquet_file)
                
                # Convert to list of dictionaries
                data = df.to_dict('records')

                # Filter valid data items
                valid_data = [item for item in data if self.validate_item(item)]
                invalid_count = len(data) - len(valid_data)

                if invalid_count > 0:
                    logger.info(f"{parquet_file.name}: filtered out {invalid_count} invalid data items, remaining {len(valid_data)} items")

                if self.test_mode:
                    target_samples = min(samples_per_file_test, len(valid_data))
                else:
                    target_samples = self.samples_per_file

                if len(valid_data) <= target_samples:
                    sampled_data = valid_data
                    logger.info(f"{parquet_file.name}: valid data insufficient ({len(valid_data)} <= {target_samples}), using all {len(valid_data)} items")
                else:
                    sampled_data = random.sample(valid_data, target_samples)
                    logger.info(f"{parquet_file.name}: randomly sampled {target_samples} items from {len(valid_data)} valid items")

                # Add source file information to each data item
                for item in sampled_data:
                    all_sampled_data.append((item, parquet_file.name))

            except Exception as e:
                logger.error(f"Error processing file {parquet_file}: {e}")
                continue

        if self.test_mode and len(all_sampled_data) > max_total_test_samples * 1.2:  
            logger.info(f"Test mode: total samples ({len(all_sampled_data)}) slightly exceeds target, but keeping all to ensure each dataset is represented")

        logger.info(f"Total sampled valid data: {len(all_sampled_data)} items")
        return all_sampled_data

    def pil_image_to_base64(self, pil_image) -> str:
        """Convert PIL Image to base64 format"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                pil_image.save(tmp_file.name, 'PNG')
                with open(tmp_file.name, "rb") as f:
                    encoded_image = base64.b64encode(f.read())
                image_base64 = f"data:image;base64,{encoded_image.decode('utf-8')}"
                os.unlink(tmp_file.name)  # Clean up temp file
                return image_base64
        except Exception as e:
            logger.error(f"Error processing PIL image: {e}")
            return None

    def prepare_messages(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare message format, remove <image> tags and process content"""
        messages = []
        images = item.get("images", [])

        # Process first user message, remove <image> tags and add image
        first_user_msg = None
        for msg in item["messages"]:
            if msg["role"] == "user":
                first_user_msg = msg
                break

        if first_user_msg and images:
            # Remove various image tags
            content = first_user_msg["content"]
            content = content.replace("<image>\n", "").replace("<image>", "")
            content = content.replace("<|image|>\n", "").replace("<|image|>", "")
            content = content.strip()

            # Skip if no content after cleaning
            if not content:
                logger.warning("No text content after cleaning, skipping data item")
                return []

            # Convert different image formats to base64
            try:
                image_base64 = self.convert_image_to_base64(images[0])
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                return []
            
            if image_base64:
                message_content = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_base64
                        },
                    },
                    {"type": "text", "text": content}
                ]
                messages.append({
                    "role": "user",
                    "content": message_content
                })
            else:
                logger.warning("Image processing failed, skipping this data item")
                return []

        return messages

    def convert_image_to_base64(self, image_data) -> str:
        """Convert various image formats to base64"""
        try:
            # 1. PIL Image object
            if hasattr(image_data, 'save'):
                return self.pil_image_to_base64(image_data)
            
            # 2. datasets Image object (usually has 'convert' method)
            elif hasattr(image_data, 'convert'):
                pil_image = image_data.convert('RGB')
                return self.pil_image_to_base64(pil_image)
            
            # 3. Dictionary format (possible datasets format)
            elif isinstance(image_data, dict):
                if 'bytes' in image_data:
                    # Create PIL Image from bytes
                    from io import BytesIO
                    pil_image = Image.open(BytesIO(image_data['bytes'])).convert('RGB')
                    return self.pil_image_to_base64(pil_image)
                elif 'path' in image_data:
                    # Load image from path
                    pil_image = Image.open(image_data['path']).convert('RGB')
                    return self.pil_image_to_base64(pil_image)
            
            # 4. numpy array
            elif hasattr(image_data, 'shape') and len(image_data.shape) in [2, 3]:
                import numpy as np
                if isinstance(image_data, np.ndarray):
                    # Ensure uint8 format
                    if image_data.dtype != np.uint8:
                        image_data = (image_data * 255).astype(np.uint8)
                    pil_image = Image.fromarray(image_data).convert('RGB')
                    return self.pil_image_to_base64(pil_image)
            
            # 5. bytes data
            elif isinstance(image_data, bytes):
                from io import BytesIO
                pil_image = Image.open(BytesIO(image_data)).convert('RGB')
                return self.pil_image_to_base64(pil_image)
            
            # 6. Try as file path
            elif isinstance(image_data, (str, Path)):
                if os.path.exists(image_data):
                    pil_image = Image.open(image_data).convert('RGB')
                    return self.pil_image_to_base64(pil_image)
            
            # 7. Last attempt: if object has __array__ method, convert to numpy array
            elif hasattr(image_data, '__array__'):
                import numpy as np
                arr = np.array(image_data)
                if len(arr.shape) in [2, 3]:
                    if arr.dtype != np.uint8:
                        arr = (arr * 255).astype(np.uint8)
                    pil_image = Image.fromarray(arr).convert('RGB')
                    return self.pil_image_to_base64(pil_image)
            
            # If no format matches, print type info for debugging
            logger.warning(f"Unknown image format: type={type(image_data)}, attributes={dir(image_data)[:5]}...")
            return None
            
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}, image_type: {type(image_data)}")
            return None

    async def request_model(self, session: aiohttp.ClientSession, messages: List[Dict[str, Any]]) -> str:
        """Async request to model, using round-robin API addresses"""
        async with self.semaphore:
            try:
                # Get next API address
                api_base = self.get_next_api_base()

                payload = {
                    "model": self.model_name,
                    "messages": messages
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

                        # Check response completeness
                        choice = result["choices"][0]
                        finish_reason = choice.get("finish_reason", "")

                        # Only responses that finished normally are considered valid
                        if finish_reason == "stop":
                            return choice["message"]["content"]
                        elif finish_reason == "length":
                            logger.warning(f"Response truncated due to max length ({api_base}), discarding this response")
                            return None
                        elif finish_reason == "content_filter":
                            logger.warning(f"Response filtered by content filter ({api_base}), discarding this response")
                            return None
                        else:
                            logger.warning(f"Unknown finish reason: {finish_reason} ({api_base}), discarding this response")
                            return None
                    else:
                        error_text = await response.text()
                        logger.error(f"API request failed ({api_base}): {response.status}, {error_text}")
                        return None
            except Exception as e:
                logger.error(f"Error requesting model: {e}")
                return None

    async def process_single_item(self, session: aiohttp.ClientSession, item_data: tuple[Dict[str, Any], str]) -> tuple[Dict[str, Any], str, str]:
        """Process single data item"""
        item, source_file = item_data

        try:
            messages = self.prepare_messages(item)

            if not messages:
                return item, source_file, None

            result = await self.request_model(session, messages)
            return item, source_file, result
        except Exception as e:
            logger.error(f"Error processing single data item: {e}")
            return item, source_file, None

    async def process_all_data(self):
        """Batch process all data"""
        # First sample all data
        all_data = self.load_and_sample_all_data()

        if not all_data:
            logger.warning("No data to process")
            return

        # Create async session
        connector = aiohttp.TCPConnector(limit=self.max_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            logger.info(f"Starting concurrent processing of {len(all_data)} data items")
            logger.info(f"Using {len(self.api_ips)} API servers: {self.api_ips}")
            logger.info(f"Maximum concurrency: {self.max_workers}")

            # Create all tasks
            tasks = [self.process_single_item(session, item_data) for item_data in all_data]

            # Execute all tasks concurrently with progress bar
            success_count = 0
            error_count = 0

            async for result in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing data"):
                try:
                    item, source_file, response = await result

                    if response is None:
                        error_count += 1
                    else:
                        # Successful result - create clean item for saving
                        processed_item = {
                            "messages": item["messages"],
                            "glm4v_response": response,
                            "source_file": source_file
                        }
                        # Remove images to save space since we don't need them in output
                        self.all_successful_data.append(processed_item)
                        success_count += 1

                except Exception as e:
                    logger.error(f"Error processing task: {e}")
                    error_count += 1

            logger.info(f"Processing completed! Success: {success_count} items, Errors discarded: {error_count} items")

    def save_all_results(self):
        """Save all merged results as parquet file"""
        if self.all_successful_data:
            output_file = Path("glm_response.parquet")

            with tqdm(total=1, desc="Saving results", unit="file") as pbar:
                # Convert to DataFrame and save as parquet
                df = pd.DataFrame(self.all_successful_data)
                df.to_parquet(output_file, index=False)
                pbar.update(1)

            logger.info(f"All results merged and saved to: {output_file}")
            logger.info(f"Total successfully processed: {len(self.all_successful_data)} data items")
        else:
            logger.warning("No successfully processed data, skipping save")

    async def run(self):
        """Run the entire processing pipeline"""
        await self.process_all_data()
        self.save_all_results()

def main():
    parser = argparse.ArgumentParser(description="Process parquet files and request GLM model")
    parser.add_argument("--data_dir", default=".",
                       help="Directory containing parquet files")
    parser.add_argument("--api_ips", nargs='+', default=["localhost"],
                       help="List of API server IPs for load balancing")
    parser.add_argument("--api_base_template", default="http://{ip}:8000/v1",
                       help="API base URL template, {ip} will be replaced with actual IP")
    parser.add_argument("--api_key", default="EMPTY", help="API key")
    parser.add_argument("--model_name", default="THUDM/GLM-4.1V-9B-Thinking", help="Model name")
    parser.add_argument("--max_workers", type=int, default=32, help="Maximum concurrency")
    parser.add_argument("--samples_per_file", type=int, default=1000, help="Number of samples per file")
    parser.add_argument("--test_mode", action="store_true", 
                       help="Enable test mode: limit to 100 samples for testing")

    args = parser.parse_args()

    processor = GLMDataProcessor(
        data_dir=args.data_dir,
        api_ips=args.api_ips,
        api_base_template=args.api_base_template,
        api_key=args.api_key,
        model_name=args.model_name,
        max_workers=args.max_workers,
        samples_per_file=args.samples_per_file,
        test_mode=args.test_mode
    )

    # Run async processing
    asyncio.run(processor.run())

if __name__ == "__main__":
    main()