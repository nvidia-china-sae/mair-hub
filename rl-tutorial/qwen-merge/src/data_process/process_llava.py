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
from datasets import load_dataset, Dataset, DownloadConfig
from datasets import Image as ImageData
from datasets.features import Sequence
from tqdm import tqdm
import traceback

AVAILABLE_DATASETS = [
    'lmms-lab/LLaVA-ReCap-558K',
    'lmms-lab/LLaVA-ReCap-CC12M',
    'lmms-lab/LLaVA-ReCap-CC3M',
]

def count_image_tags(text):
    """Count the number of <image> tags in text"""
    return text.count('<image>')

def has_video_tag(text):
    """Check if text contains <video> tags"""
    return '<video>' in text.lower()

def has_think_tags(text):
    """Check if text contains <think> tags"""
    return '<think>' in text.lower() or '</think>' in text.lower()

def has_special_media_tags(text):
    """Check if text contains special media tags"""
    special_tags = ['<image>', '<audio>', '<video>']
    for tag in special_tags:
        if tag in text.lower():
            return True
    return False

def process_single_item(item):
    """Process a single data item for data.map()"""
    try:
        if 'image' not in item or item['image'] is None:
            return {'messages': None, 'images': None}

        if 'conversations' not in item or not item['conversations']:
            return {'messages': None, 'images': None}

        conversations = item['conversations']

        if len(conversations) % 2 != 0:
            return {'messages': None, 'images': None}

        for i, conv in enumerate(conversations):
            expected_from = 'human' if i % 2 == 0 else 'gpt'
            if conv['from'] != expected_from:
                return {'messages': None, 'images': None}

        conversations = conversations.copy()
        for i, conv in enumerate(conversations):
            if conv['from'] == 'human':
                user_content = conv['value']

                if i == 0:
                    image_count = count_image_tags(user_content)
                    if image_count == 0:
                        conversations[0] = conversations[0].copy()
                        conversations[0]['value'] = '<image>\n' + user_content
                    elif image_count > 1:
                        return {'messages': None, 'images': None}

                    if has_video_tag(user_content):
                        return {'messages': None, 'images': None}
                else:
                    if has_special_media_tags(user_content):
                        return {'messages': None, 'images': None}

            elif conv['from'] == 'gpt':
                if has_special_media_tags(conv['value']):
                    return {'messages': None, 'images': None}

                if has_think_tags(conv['value']):
                    return {'messages': None, 'images': None}

        messages = []
        for conv in conversations:
            if conv['from'] == 'human':
                messages.append({
                    "content": conv['value'],
                    "role": "user"
                })
            elif conv['from'] == 'gpt':
                messages.append({
                    "content": conv['value'],
                    "role": "assistant"
                })

        return {
            "messages": messages,
            "images": [item['image']]
        }

    except Exception as e:
        return {'messages': None, 'images': None}

def save_dataset_simple(valid_data, dataset_name, output_base_dir):
    """Save dataset as single parquet file"""
    clean_dataset_name = dataset_name.replace('lmms-lab/', '').replace('-', '_')
    output_file = os.path.join(output_base_dir, f"{clean_dataset_name}.parquet")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Saving converted data to: {output_file}")
    valid_data.to_parquet(output_file)
    return output_file

def process_dataset_with_map(dataset_name, output_base_dir="./llava_recap_converted", num_processes=None, test_mode=False, max_samples=2000000):
    """Process dataset using data.map() method"""
    print(f"Starting to process dataset: {dataset_name}")
    if test_mode:
        print("Running in test mode - will process maximum 10,000 samples per dataset")

    download_config = DownloadConfig(
        max_retries=5,
        num_proc=8,
        resume_download=True,
    )

    print("Loading full dataset with enhanced download configuration...")
    dataset = load_dataset(dataset_name, download_config=download_config)

    data = dataset['train']
    total_length = len(data)
    print(f"Dataset total length: {total_length}")

    if test_mode and total_length > 1000:
        print(f"Test mode: Limiting dataset to 10,000 samples for processing (original: {total_length})")
        data = data.select(range(1000))
        total_length = len(data)

    print("Starting concurrent data processing...")
    processed_data = data.map(
        process_single_item,
        num_proc=num_processes,
        desc="Processing data items",
        load_from_cache_file=False,
        remove_columns=data.column_names
    )

    print("Filtering valid data...")

    valid_data = processed_data.filter(
        lambda x: x['messages'] is not None and x['images'] is not None,
        num_proc=num_processes,
        desc="Filtering valid data"
    )

    original_valid_count = len(valid_data)
    if original_valid_count > max_samples:
        print(f"Dataset size ({original_valid_count}) exceeds maximum limit, limiting to {max_samples} samples")
        valid_data = valid_data.select(range(max_samples))

    print("Converting image column format...")
    valid_data = valid_data.cast_column('images', Sequence(feature=ImageData(decode=True)))

    valid_count = len(valid_data)
    stats = {
        'total': total_length,
        'valid': valid_count,
        'filtered': total_length - valid_count,
        'limited': original_valid_count - valid_count if original_valid_count > max_samples else 0
    }

    saved_file = save_dataset_simple(valid_data, dataset_name, output_base_dir)

    test_mode_info = " (Test mode: processed 10K samples)" if test_mode else ""
    limit_info = f" (Limited to {max_samples} from {original_valid_count})" if original_valid_count > max_samples else ""
    print(f"""
Dataset {dataset_name} processing completed{test_mode_info}{limit_info}:
  - Original data count: {stats['total']}
  - Valid data: {stats['valid']} ({stats['valid']/stats['total']*100:.1f}%)
  - Filtered out: {stats['filtered']}
  - Limited (if applicable): {stats['limited']}
  - Saved to: {saved_file}
""")

    return valid_data, stats

def process_all_datasets_with_map(output_base_dir="./llava_recap_converted",
                                 datasets_to_process=None,
                                 num_processes=None,
                                 test_mode=False,
                                 max_samples=2000000):
    """Process all datasets using data.map() method"""

    os.makedirs(output_base_dir, exist_ok=True)

    if datasets_to_process is None:
        datasets_to_process = AVAILABLE_DATASETS

    mode_info = " (Test Mode)" if test_mode else ""
    print(f"LLaVA-ReCap Dataset Processing - data.map() version{mode_info}:")
    print(f"Preparing to process {len(datasets_to_process)} dataset(s)")
    if test_mode:
        print("Test mode enabled: Each dataset will be limited to 10,000 samples")
    if num_processes:
        print(f"Using {num_processes} processes")
    else:
        print("Using all available CPU cores")
    print(f"Output directory: {output_base_dir}")
    print(f"Maximum samples per dataset: {max_samples}")

    all_converted_data = []
    total_stats = {
        'total': 0,
        'valid': 0,
        'filtered': 0
    }

    for dataset_name in datasets_to_process:
        print(f"\n{'='*80}")
        print(f"Starting to process dataset: {dataset_name}")

        try:
            valid_data, stats = process_dataset_with_map(
                dataset_name, output_base_dir, num_processes, test_mode, max_samples
            )

            all_converted_data.extend(list(valid_data))

            for key in ['total', 'valid', 'filtered']:
                if key in stats:
                    total_stats[key] += stats[key]

        except Exception as e:
            print(f"Skipping dataset {dataset_name}: {e}")
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    final_mode_info = " (Test Mode)" if test_mode else ""
    print(f"Final processing statistics{final_mode_info}:")
    for key, value in total_stats.items():
        if key == 'total':
            print(f"{key}: {value}")
        elif key == 'valid':
            print(f"{key}: {value} ({value/max(total_stats['total'], 1)*100:.1f}%)")
        else:
            print(f"{key}: {value}")
    print(f"{'='*80}")

    return all_converted_data

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Process LLaVA-ReCap datasets")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./llava_recap_converted",
        help="Output directory for converted datasets (default: ./llava_recap_converted)"
    )
    parser.add_argument(
        "--num_processes", 
        type=int, 
        default=None,
        help="Number of processes for parallel processing (default: use all CPU cores)"
    )
    parser.add_argument(
        "--datasets", 
        type=str, 
        nargs='+',
        default=None,
        help="Specific datasets to process (default: process all available datasets)"
    )
    parser.add_argument(
        "--test_mode", 
        action="store_true",
        help="Enable test mode: limit each dataset to 10,000 samples"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=2000000,
        help="Maximum number of samples per dataset (default: 2000000)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    print("LLaVA-ReCap Dataset Processing - data.map() version")
    print("Features: Uses datasets built-in concurrency, more efficient memory management, simplified single-file saving")

    args = parse_args()
    
    print(f"Configuration:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Number of processes: {args.num_processes if args.num_processes else 'all CPU cores'}")
    print(f"  Maximum samples per dataset: {args.max_samples}")
    if args.datasets:
        print(f"  Datasets to process: {args.datasets}")
    else:
        print(f"  Datasets to process: Using predefined configurations")
        for name in AVAILABLE_DATASETS:
            print(f"    - {name}")
    print(f"  Test mode: {args.test_mode}")

    converted_data = process_all_datasets_with_map(
        output_base_dir=args.output_dir,
        datasets_to_process=args.datasets,
        num_processes=args.num_processes,
        test_mode=args.test_mode,
        max_samples=args.max_samples
    )

    print(f"\nProcessing completed! Total converted data: {len(converted_data)} items")