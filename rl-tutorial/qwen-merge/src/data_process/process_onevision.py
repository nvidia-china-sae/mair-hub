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

# List of all available LLaVA-OneVision configs
AVAILABLE_CONFIGS = [
    'CLEVR-Math(MathV360K)', 'Evol-Instruct-GPT4-Turbo', 'FigureQA(MathV360K)',
    'GEOS(MathV360K)', 'GeoQA+(MathV360K)', 'Geometry3K(MathV360K)',
    'IconQA(MathV360K)', 'MapQA(MathV360K)', 'MathV360K_TQA', 'MathV360K_VQA-AS',
    'MathV360K_VQA-RAD', 'PMC-VQA(MathV360K)', 'Super-CLEVR(MathV360K)',
    'TabMWP(MathV360K)', 'UniGeo(MathV360K)', 'VisualWebInstruct(filtered)',
    'VizWiz(MathV360K)', 'ai2d(cauldron,llava_format)', 'ai2d(gpt4v)',
    'ai2d(internvl)', 'allava_instruct_laion4v', 'allava_instruct_vflan4v',
    'aokvqa(cauldron,llava_format)', 'chart2text(cauldron)',
    'chartqa(cauldron,llava_format)', 'chrome_writting', 'clevr(cauldron,llava_format)',
    'diagram_image_to_text(cauldron)', 'dvqa(cauldron,llava_format)',
    'figureqa(cauldron,llava_format)', 'geo170k(align)', 'geo170k(qa)', 'geo3k',
    'geomverse(cauldron)', 'hateful_memes(cauldron,llava_format)',
    'hitab(cauldron,llava_format)', 'hme100k', 'iam(cauldron)',
    'iconqa(cauldron,llava_format)', 'iiit5k', 'image_textualization(filtered)',
    'infographic(gpt4v)', 'infographic_vqa', 'infographic_vqa_llava_format',
    'intergps(cauldron,llava_format)', 'k12_printing', 'llava_wild_4v_12k_filtered',
    'llava_wild_4v_39k_filtered', 'llavar_gpt4_20k', 'lrv_chart',
    'lrv_normal(filtered)', 'magpie_pro(l3_80b_mt)', 'magpie_pro(l3_80b_st)',
    'magpie_pro(qwen2_72b_st)', 'mapqa(cauldron,llava_format)', 'mathqa',
    'mavis_math_metagen', 'mavis_math_rule_geo', 'multihiertt(cauldron)',
    'orand_car_a', 'raven(cauldron)', 'rendered_text(cauldron)',
    'robut_sqa(cauldron)', 'robut_wikisql(cauldron)', 'robut_wtq(cauldron,llava_format)',
    'scienceqa(cauldron,llava_format)', 'scienceqa(nona_context)',
    'screen2words(cauldron)', 'sharegpt4o', 'sharegpt4v(coco)',
    'sharegpt4v(knowledge)', 'sharegpt4v(llava)', 'sharegpt4v(sam)', 'sroie',
    'st_vqa(cauldron,llava_format)', 'tabmwp(cauldron)', 'tallyqa(cauldron,llava_format)',
    'textcaps', 'textocr(gpt4v)', 'tqa(cauldron,llava_format)', 'ureader_cap',
    'ureader_ie', 'vision_flan(filtered)', 'vistext(cauldron)',
    'visual7w(cauldron,llava_format)', 'visualmrc(cauldron)',
    'vqarad(cauldron,llava_format)', 'vsr(cauldron,llava_format)',
    'websight(cauldron)'
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
    """
    Process a single data item for data.map()
    Only returns required fields: messages and images
    """
    try:
        # Check if image field exists and is not None
        if 'image' not in item or item['image'] is None:
            return {'messages': None, 'images': None}

        # Check conversations field
        if 'conversations' not in item or not item['conversations']:
            return {'messages': None, 'images': None}

        conversations = item['conversations']

        # Check if conversation format is correct
        if len(conversations) % 2 != 0:
            return {'messages': None, 'images': None}

        # Check conversation format (should be human, gpt, human, gpt, ...)
        for i, conv in enumerate(conversations):
            expected_from = 'human' if i % 2 == 0 else 'gpt'
            if conv['from'] != expected_from:
                return {'messages': None, 'images': None}

        # Check all messages
        conversations = conversations.copy()
        for i, conv in enumerate(conversations):
            if conv['from'] == 'human':
                user_content = conv['value']

                # Special handling for first human message
                if i == 0:
                    # Check if dataset contains cauldron or llava_format
                    is_special_dataset = 'cauldron' in str(item.get('config_name', '')).lower() or 'llava_format' in str(item.get('config_name', '')).lower()

                    if is_special_dataset:
                        # For special datasets, add <image> tag if not present
                        image_count = count_image_tags(user_content)
                        if image_count == 0:
                            conversations[0] = conversations[0].copy()
                            conversations[0]['value'] = '<image>\n' + user_content
                        elif image_count > 1:
                            return {'messages': None, 'images': None}
                    else:
                        # For normal datasets, check <image> tag count (should be exactly 1)
                        image_count = count_image_tags(user_content)
                        if image_count != 1:
                            return {'messages': None, 'images': None}

                    # Check if contains <video> tag
                    if has_video_tag(user_content):
                        return {'messages': None, 'images': None}
                else:
                    # Non-first human messages: cannot contain any special media tags
                    if has_special_media_tags(user_content):
                        return {'messages': None, 'images': None}

            elif conv['from'] == 'gpt':
                # Check assistant messages: cannot contain any special media tags
                if has_special_media_tags(conv['value']):
                    return {'messages': None, 'images': None}

                # Check if answer contains <think> tags
                if has_think_tags(conv['value']):
                    return {'messages': None, 'images': None}

        # Convert to target format
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

        # Return final format
        return {
            "messages": messages,
            "images": [item['image']]
        }

    except Exception as e:
        return {'messages': None, 'images': None}

def clean_config_name(config_name):
    """Clean config name and remove trailing underscores"""
    # Remove lmms-lab/ prefix if exists
    clean_name = config_name.replace('lmms-lab/', '')
    # Replace special characters with underscores
    clean_name = clean_name.replace('(', '_').replace(')', '_').replace(',', '_').replace(' ', '_').replace('-', '_')
    # Remove trailing underscores
    clean_name = clean_name.rstrip('_')
    return clean_name

def save_dataset_simple(valid_data, config_name, output_base_dir):
    """
    Save dataset as single parquet file
    """
    if len(valid_data) == 0:
        print(f"Skipping save: dataset {config_name} has no valid data")
        return None
    
    clean_dataset_name = clean_config_name(config_name)
    output_file = os.path.join(output_base_dir, f"{clean_dataset_name}.parquet")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Saving converted data to: {output_file}")
    valid_data.to_parquet(output_file)
    return output_file

def process_dataset_with_map(config_name, output_base_dir="./llavaonevision_converted", num_processes=None, test_mode=False):
    """
    Process dataset using data.map() method
    """
    print(f"Starting to process dataset: {config_name}")
    if test_mode:
        print("Running in test mode - will process maximum 100 samples per dataset")

    # Create download configuration
    download_config = DownloadConfig(
        max_retries=5,
        num_proc=8,
        resume_download=True,
    )

    # Load dataset
    print("Loading dataset with enhanced download configuration...")
    dataset = load_dataset("lmms-lab/LLaVA-OneVision-Data", config_name, download_config=download_config)

    data = dataset['train']
    total_length = len(data)
    print(f"Dataset total length: {total_length}")

    # Apply test mode limit early
    if test_mode and total_length > 100:
        print(f"Test mode: Limiting dataset to 100 samples for processing (original: {total_length})")
        data = data.select(range(100))
        total_length = len(data)

    # Add config_name to each item for special dataset detection
    def add_config_name(item):
        item['config_name'] = config_name
        return item

    data = data.map(add_config_name, desc="Adding config name")

    # Process data using map
    print("Starting concurrent data processing...")
    processed_data = data.map(
        process_single_item,
        num_proc=num_processes,
        desc="Processing data items",
        load_from_cache_file=False,
        remove_columns=data.column_names
    )

    print("Filtering valid data...")

    # Filter valid data
    valid_data = processed_data.filter(
        lambda x: x['messages'] is not None and x['images'] is not None,
        num_proc=num_processes,
        desc="Filtering valid data"
    )

    # Convert image column format before saving
    print("Converting image column format...")
    valid_data = valid_data.cast_column('images', Sequence(feature=ImageData(decode=True)))

    # Statistics
    valid_count = len(valid_data)
    stats = {
        'total': total_length,
        'valid': valid_count,
        'filtered': total_length - valid_count
    }

    # Save results as single Parquet file
    saved_file = save_dataset_simple(valid_data, config_name, output_base_dir)

    # Print statistics
    test_mode_info = " (Test mode: processed 100 samples)" if test_mode else ""
    if saved_file:
        result_info = f"  - Saved to: {saved_file}"
    else:
        result_info = "  - Not saved: no valid data"
    
    print(f"""
Dataset {config_name} processing completed{test_mode_info}:
  - Original data count: {stats['total']}
  - Valid data: {stats['valid']} ({stats['valid']/stats['total']*100:.1f}%)
  - Filtered out: {stats['filtered']}
{result_info}
""")

    return valid_data, stats

def process_all_datasets_with_map(output_base_dir="./llavaonevision_converted",
                                 configs_to_process=None,
                                 num_processes=None,
                                 test_mode=False):
    """
    Process all datasets using data.map() method
    """

    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)

    # Determine configs to process
    if configs_to_process is None:
        configs_to_process = AVAILABLE_CONFIGS

    mode_info = " (Test Mode)" if test_mode else ""
    print(f"LLaVA-OneVision Dataset Processing - data.map() version{mode_info}:")
    print(f"Preparing to process {len(configs_to_process)} dataset(s)")
    if test_mode:
        print("Test mode enabled: Each dataset will be limited to 100 samples")
    if num_processes:
        print(f"Using {num_processes} processes")
    else:
        print("Using all available CPU cores")
    print(f"Output directory: {output_base_dir}")

    # Process all datasets
    all_converted_data = []
    total_stats = {
        'total': 0,
        'valid': 0,
        'filtered': 0
    }

    for config_name in configs_to_process:
        print(f"\n{'='*80}")
        print(f"Starting to process dataset: {config_name}")

        try:
            valid_data, stats = process_dataset_with_map(
                config_name, output_base_dir, num_processes, test_mode
            )

            # Convert Dataset to list and add to total list
            all_converted_data.extend(list(valid_data))

            # Accumulate statistics
            for key in ['total', 'valid', 'filtered']:
                if key in stats:
                    total_stats[key] += stats[key]

        except Exception as e:
            print(f"Skipping dataset {config_name}: {e}")
            traceback.print_exc()
            continue

    # Final statistics
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
    parser = argparse.ArgumentParser(description="Process LLaVA-OneVision datasets")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./llavaonevision_converted",
        help="Output directory for converted datasets (default: ./llavaonevision_converted)"
    )
    parser.add_argument(
        "--num_processes", 
        type=int, 
        default=None,
        help="Number of processes for parallel processing (default: use all CPU cores)"
    )
    parser.add_argument(
        "--configs", 
        type=str, 
        nargs='+',
        default=None,
        help="Specific configs to process (default: process all available configs)"
    )
    parser.add_argument(
        "--test_mode", 
        action="store_true",
        help="Enable test mode: limit each dataset to 100 samples"
    )
    return parser.parse_args()

if __name__ == "__main__":
    print("LLaVA-OneVision Dataset Processing - data.map() version")
    print("Features: Uses datasets built-in concurrency, more efficient memory management, parquet format saving")

    args = parse_args()
    
    print(f"Configuration:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Number of processes: {args.num_processes if args.num_processes else 'all CPU cores'}")
    if args.configs:
        print(f"  Configs to process: {args.configs}")
    else:
        print(f"  Configs to process: Using predefined configurations")
        for name in AVAILABLE_CONFIGS:
            print(f"    - {name}")
    print(f"  Test mode: {args.test_mode}")

    converted_data = process_all_datasets_with_map(
        output_base_dir=args.output_dir,
        configs_to_process=args.configs,
        num_processes=args.num_processes,
        test_mode=args.test_mode
    )

    print(f"\nProcessing completed! Total converted data: {len(converted_data)} items")