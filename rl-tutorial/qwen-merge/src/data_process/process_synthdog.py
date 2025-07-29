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
import argparse
from typing import List, Optional, Union
from datasets import load_dataset, Dataset, DownloadConfig
from datasets import Image as ImageData
from datasets.features import Sequence
import traceback

LANGUAGE_CONFIGS = {
    "en": {
        "dataset_name": "naver-clova-ix/synthdog-en",
        "user_prompt": "<image>\nPlease recognize all text in the image and output the text.",
        "output_suffix": "en"
    },
    "zh": {
        "dataset_name": "naver-clova-ix/synthdog-zh", 
        "user_prompt": "<image>\nPlease identify the text in this image and output plain text.",
        "output_suffix": "zh"
    }
}

def create_process_function(language: str):
    """Create data processing function for specified language"""
    config = LANGUAGE_CONFIGS[language]
    user_prompt = config["user_prompt"]
    
    def process_single_item(item):
        """Convert a synthdog sample to required {messages, images} format"""
        image = item.get("image")
        if image is None:
            return {"messages": None, "images": None}

        try:
            if not hasattr(image, 'size') or not image.size:
                return {"messages": None, "images": None}
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > 50000000:  # 50M pixels limit
                return {"messages": None, "images": None}
        except Exception:
            return {"messages": None, "images": None}

        try:
            gt = eval(item["ground_truth"])
            text_content = gt["gt_parse"]["text_sequence"]
            if not text_content or not isinstance(text_content, str) or len(text_content.strip()) == 0:
                return {"messages": None, "images": None}
        except Exception:
            return {"messages": None, "images": None}

        messages = [
            {
                "content": user_prompt,
                "role": "user",
            },
            {
                "content": text_content.strip(),
                "role": "assistant",
            },
        ]

        return {"messages": messages, "images": [image]}
    
    return process_single_item

def is_valid_item(x):
    """Validate data item"""
    if x["messages"] is None or x["images"] is None:
        return False
    if not isinstance(x["images"], list) or len(x["images"]) == 0:
        return False
    try:
        img = x["images"][0]
        if not hasattr(img, 'size') or not img.size:
            return False
        return True
    except Exception:
        return False

def process_single_dataset(
    language: str,
    output_base_dir: str,
    num_processes: Optional[int] = None,
    test_mode: bool = False
):
    """Process single language dataset"""
    config = LANGUAGE_CONFIGS[language]
    dataset_name = config["dataset_name"]
    output_suffix = config["output_suffix"]
    
    print(f"Starting to process {language.upper()} dataset: {dataset_name}")
    if test_mode:
        print("Running in test mode - will process maximum 10,000 samples")
    
    download_config = DownloadConfig(
        max_retries=5,
        num_proc=8,
        resume_download=True,
    )
    
    print("Loading dataset with enhanced download configuration...")
    dataset = load_dataset(dataset_name, download_config=download_config)
    data = dataset["train"]
    total_len = len(data)
    print(f"Dataset total length: {total_len}")

    if test_mode and total_len > 1000:
        print(f"Test mode: Limiting dataset to 1,000 samples (original: {total_len})")
        data = data.select(range(1000))
        total_len = len(data)

    process_func = create_process_function(language)
    processed = data.map(
        process_func,
        num_proc=num_processes,
        desc=f"Processing {language.upper()} data items",
        load_from_cache_file=False,
        remove_columns=data.column_names,
    )

    print(f"Filtering valid {language.upper()} data...")
    valid_data = processed.filter(
        is_valid_item,
        num_proc=num_processes,
        desc=f"Filtering {language.upper()} valid data",
    )

    print(f"Converting {language.upper()} image data format...")
    try:
        valid_data = valid_data.cast_column("images", Sequence(feature=ImageData(decode=True)))
    except Exception as e:
        print(f"Image format conversion error: {e}")
        print("Re-filtering data...")
        valid_data = valid_data.filter(
            lambda x: x["images"] is not None and isinstance(x["images"], list) and len(x["images"]) > 0,
            desc="Re-filtering data"
        )
        valid_data = valid_data.cast_column("images", Sequence(feature=ImageData(decode=True)))

    os.makedirs(output_base_dir, exist_ok=True)
    output_path = os.path.join(output_base_dir, f"synthdog_{output_suffix}.parquet")
    print(f"Saving {language.upper()} converted data to: {output_path}")
    valid_data.to_parquet(output_path)

    valid_count = len(valid_data)
    test_mode_info = " (Test mode)" if test_mode else ""
    print(f"""
{language.upper()} dataset processing completed{test_mode_info}:
  - Original data count: {total_len}
  - Valid data: {valid_count} ({valid_count / total_len * 100:.1f}%)
  - Filtered out: {total_len - valid_count}
""")
    return valid_data, {"total": total_len, "valid": valid_count, "filtered": total_len - valid_count}

def process_synthdog_datasets(
    languages: Union[str, List[str]] = ["en", "zh"],
    output_base_dir: str = "./synthdog_converted",
    num_processes: Optional[int] = None,
    test_mode: bool = False,
):
    """Process synthdog datasets for specified languages"""
    if isinstance(languages, str):
        languages = [languages]
    
    mode_info = " (Test Mode)" if test_mode else ""
    print(f"SynthDog Dataset Processing{mode_info}:")
    print(f"Languages to process: {languages}")
    if test_mode:
        print("Test mode enabled: Each dataset will be limited to 10,000 samples")
    if num_processes:
        print(f"Using {num_processes} processes")
    else:
        print("Using all available CPU cores")
    print(f"Output directory: {output_base_dir}")
    
    processed_datasets = []
    total_stats = {"total": 0, "valid": 0, "filtered": 0}
    
    for lang in languages:
        if lang not in LANGUAGE_CONFIGS:
            print(f"Warning: Unsupported language '{lang}', skipping")
            continue
        
        print(f"\n{'='*80}")
        try:
            dataset, stats = process_single_dataset(
                language=lang,
                output_base_dir=output_base_dir,
                num_processes=num_processes,
                test_mode=test_mode
            )
            processed_datasets.append(dataset)
            
            for key in total_stats.keys():
                total_stats[key] += stats[key]
                
        except Exception as e:
            print(f"Error processing {lang} dataset: {e}")
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"Final processing statistics{mode_info}:")
    for key, value in total_stats.items():
        if key == 'total':
            print(f"{key}: {value}")
        elif key == 'valid':
            print(f"{key}: {value} ({value/max(total_stats['total'], 1)*100:.1f}%)")
        else:
            print(f"{key}: {value}")
    print(f"{'='*80}")
    
    return processed_datasets

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Process SynthDog datasets")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./synthdog_converted",
        help="Output directory for converted datasets (default: ./synthdog_converted)"
    )
    parser.add_argument(
        "--num_processes", 
        type=int, 
        default=None,
        help="Number of processes for parallel processing (default: use all CPU cores)"
    )
    parser.add_argument(
        "--languages", 
        type=str, 
        nargs='+',
        default=["en", "zh"],
        choices=["en", "zh"],
        help="Languages to process (default: ['en', 'zh'])"
    )
    parser.add_argument(
        "--test_mode", 
        action="store_true",
        help="Enable test mode: limit each dataset to 10,000 samples"
    )
    return parser.parse_args()

if __name__ == "__main__":
    print("SynthDog Dataset Processing")
    print("Features: Multi-language OCR dataset conversion with concurrent processing")

    args = parse_args()
    
    print(f"Configuration:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Number of processes: {args.num_processes if args.num_processes else 'all CPU cores'}")
    print(f"  Languages to process: {args.languages}")
    print(f"  Test mode: {args.test_mode}")

    converted_data = process_synthdog_datasets(
        languages=args.languages,
        output_base_dir=args.output_dir,
        num_processes=args.num_processes,
        test_mode=args.test_mode
    )

    total_count = sum(len(dataset) for dataset in converted_data)
    print(f"\nProcessing completed! Total converted data: {total_count} items")