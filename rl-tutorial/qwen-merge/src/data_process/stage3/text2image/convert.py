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

from PIL import Image, ImageDraw, ImageFont
import random
import colorsys
import os
import json
import argparse
import re
from tqdm import tqdm
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from datasets import Dataset
from datasets import Image as ImageData
from datasets.features import Sequence

_cached_font = None
_font_detection_done = False

def generate_contrasting_colors():
    """Generate a pair of high contrast colors"""
    hue = random.random()
    bg_saturation = random.uniform(0.1, 0.3)
    bg_value = random.uniform(0.9, 1.0)
    bg_rgb = colorsys.hsv_to_rgb(hue, bg_saturation, bg_value)
    bg_color = tuple(int(x * 255) for x in bg_rgb)

    text_hue = (hue + 0.5) % 1.0
    text_saturation = random.uniform(0.7, 1.0)
    text_value = random.uniform(0.3, 0.7)
    text_rgb = colorsys.hsv_to_rgb(text_hue, text_saturation, text_value)
    text_color = tuple(int(x * 255) for x in text_rgb)

    return bg_color, text_color

def detect_and_cache_font(font_size, silent=False):
    """Detect and cache Chinese-compatible font (execute only once)"""
    global _cached_font, _font_detection_done

    if _font_detection_done:
        return _cached_font

    if not silent:
        print("=" * 50)
        print("Detecting Chinese-compatible fonts...")

    # More comprehensive Chinese font path list
    chinese_font_paths = [
        # WenQuanYi fonts (most common Linux Chinese fonts)
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
        "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",

        # Noto fonts
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",

        # AR PL fonts
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/arphic-uming/uming.ttc",

        # DroidSans (Android fonts)
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/droid/DroidSansFallbackFull.ttf",

        # CentOS/RHEL Chinese fonts
        "/usr/share/fonts/chinese/TrueType/uming.ttf",
        "/usr/share/fonts/chinese/TrueType/ukai.ttf",

        # Other possible paths
        "/usr/share/fonts/TTF/wqy-microhei.ttc",
        "/usr/share/fonts/OTF/wqy-microhei.ttc",

        # macOS Chinese fonts
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",

        # Windows Chinese fonts
        "C:\\Windows\\Fonts\\msyh.ttc",     # Microsoft YaHei
        "C:\\Windows\\Fonts\\simsun.ttc",   # SimSun
        "C:\\Windows\\Fonts\\simhei.ttf",   # SimHei
    ]

    # Store found font path
    found_font_path = None

    # Test each font path
    for font_path in chinese_font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)

                # Test Chinese rendering
                test_img = Image.new('RGB', (200, 100))
                test_draw = ImageDraw.Draw(test_img)
                test_text = "测试中文字体"

                try:
                    bbox = test_draw.textbbox((0, 0), test_text, font=font)
                    test_draw.text((10, 10), test_text, font=font, fill="black")

                    # Check if bbox is reasonable
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                        if not silent:
                            print(f"Found and using Chinese font: {os.path.basename(font_path)}")
                        _cached_font = font
                        found_font_path = font_path
                        _font_detection_done = True
                        if not silent:
                            print("=" * 50)
                        return font, found_font_path

                except Exception:
                    continue

            except Exception:
                continue

    # Try loading by font name
    generic_font_names = [
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Noto Sans CJK",
        "Droid Sans Fallback",
        "AR PL UMing CN",
        "SimSun",
        "SimHei",
        "Microsoft YaHei",
    ]

    for font_name in generic_font_names:
        try:
            font = ImageFont.truetype(font_name, font_size)

            # Test Chinese rendering
            test_img = Image.new('RGB', (200, 100))
            test_draw = ImageDraw.Draw(test_img)
            test_text = "测试中文"

            try:
                bbox = test_draw.textbbox((0, 0), test_text, font=font)
                test_draw.text((10, 10), test_text, font=font, fill="black")

                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    if not silent:
                        print(f"Found Chinese font by name: {font_name}")
                    _cached_font = font
                    found_font_path = font_name
                    _font_detection_done = True
                    if not silent:
                        print("=" * 50)
                    return font, found_font_path

            except Exception:
                continue

        except Exception:
            continue

    # If all Chinese fonts fail
    if not silent:
        print("Warning: No Chinese-compatible font found!")
        print("Chinese characters may display as boxes. Recommend installing Chinese fonts:")
        print("Ubuntu/Debian: sudo apt-get install fonts-wqy-microhei")
        print("CentOS/RHEL: sudo yum install wqy-microhei-fonts")

    # Use fallback font
    try:
        fallback_font = ImageFont.truetype("DejaVu Sans", font_size)
        if not silent:
            print("Using DejaVu Sans as fallback font")
        _cached_font = fallback_font
        found_font_path = "DejaVu Sans"
        _font_detection_done = True
        if not silent:
            print("=" * 50)
        return fallback_font, found_font_path
    except:
        if not silent:
            print("Using system default font")
        _cached_font = ImageFont.load_default()
        found_font_path = "default"
        _font_detection_done = True
        if not silent:
            print("=" * 50)
        return _cached_font, found_font_path

def get_font(font_size):
    """Get font (using cache, avoid repeated detection)"""
    global _cached_font, _font_detection_done

    if not _font_detection_done:
        font, _ = detect_and_cache_font(font_size)
        return font

    # If font size changes, need to recreate font object
    if _cached_font and hasattr(_cached_font, 'path'):
        try:
            return ImageFont.truetype(_cached_font.path, font_size)
        except:
            pass

    return _cached_font

def create_sample_image(text, image_type):
    """Create a sample image with text"""
    margin = 40
    font_size = 24
    line_height = 32
    width = 600
    max_height = 400

    font = get_font(font_size)
    
    image = Image.new('RGB', (width, max_height))
    draw = ImageDraw.Draw(image)

    bg_color, text_color = generate_contrasting_colors()
    image.paste(bg_color, (0, 0, width, max_height))

    is_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
    
    lines = []
    if is_chinese:
        current_line = ""
        for char in text:
            test_line = current_line + char
            try:
                bbox = draw.textbbox((0, 0), test_line, font=font)
                text_width = bbox[2] - bbox[0]
            except:
                text_width = len(test_line) * font_size * 0.6
                
            if text_width < width - 2 * margin:
                current_line += char
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = char
                else:
                    break
        if current_line:
            lines.append(current_line)
    else:
        words = text.split()
        current_line = []
        for word in words:
            test_line = ' '.join(current_line + [word])
            try:
                bbox = draw.textbbox((0, 0), test_line, font=font)
                text_width = bbox[2] - bbox[0]
            except:
                text_width = len(test_line) * font_size * 0.5
                
            if text_width < width - 2 * margin:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    break
        if current_line:
            lines.append(' '.join(current_line))

    y = margin
    for line in lines:
        if y + line_height > max_height - margin:
            break
        try:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
        except:
            text_width = len(line) * font_size * 0.6
            
        x = (width - text_width) // 2
        draw.text((x, y), line, font=font, fill=text_color)
        y += line_height

    return image

def detect_english(text):
    """Detect if text is primarily English"""
    letters_only = re.sub(r'[^a-zA-Z\u4e00-\u9fff]', '', text)
    if not letters_only:
        return False

    english_chars = len(re.findall(r'[a-zA-Z]', letters_only))
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', letters_only))
    total_chars = english_chars + chinese_chars

    if total_chars == 0:
        return False

    english_ratio = english_chars / total_chars
    chinese_ratio = chinese_chars / total_chars

    return english_ratio > 0.5 and chinese_ratio <= 0.05

def create_user_message(query, mode):
    """Create user message based on query language and mode"""
    is_english = detect_english(query)

    if is_english:
        base_message = "<image>\nPlease answer the question in the image."
        if mode == "no_thinking":
            return base_message + " /no_think"
        return base_message
    else:
        base_message = "<image>\n请回答图片中的问题。"
        if mode == "no_thinking":
            return base_message + " /no_think"
        return base_message

def process_single_item(item):
    """Process a single data item for datasets.map()"""
    try:
        query = item['query']
        response = item['response']
        mode = item['mode']
        query_index = item.get('query_index', 0)
        model = item.get('model', 'unknown')

        if '<image>' in response.lower() or '<video>' in response.lower() or '<audio>' in response.lower():
            return {'messages': None, 'images': None}

        is_chinese = not detect_english(query)
        sample_text = "This is a sample English question for text-to-image conversion." if not is_chinese else "这是一个中文文本转图像的示例问题。"
        image_type = "chinese" if is_chinese else "english"
        
        image = create_sample_image(sample_text, image_type)
        
        user_message = create_user_message(query, mode)

        messages = [
            {
                "content": user_message,
                "role": "user"
            },
            {
                "content": response,
                "role": "assistant"
            }
        ]

        return {
            "messages": messages,
            "images": [image]
        }

    except Exception as e:
        return {'messages': None, 'images': None}

def convert_to_multimodal_format_with_map(input_files, output_file, max_workers=None):
    """
    Convert process.py generated data to multimodal conversation format
    Using datasets.map() method similar to process_llava.py
    """
    print("Starting data conversion...")

    detect_and_cache_font(24)
    
    print("Creating sample images...")
    chinese_sample = create_sample_image("这是一个中文文本转图像的示例问题。", "chinese")
    english_sample = create_sample_image("This is a sample English question for text-to-image conversion.", "english")
    
    os.makedirs("sample_images", exist_ok=True)
    chinese_sample.save("sample_images/chinese_sample.png")
    english_sample.save("sample_images/english_sample.png")
    print("Sample images saved to sample_images/")

    all_items = []
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"Warning: File does not exist {input_file}")
            continue

        print(f"Loading file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_items.extend(data)

    total_items = len(all_items)
    print(f"Total conversations to process: {total_items}")

    dataset = Dataset.from_list(all_items)
    
    print("Starting concurrent data processing...")
    processed_data = dataset.map(
        process_single_item,
        num_proc=max_workers,
        desc="Processing data items",
        load_from_cache_file=False,
        remove_columns=dataset.column_names
    )

    print("Filtering valid data...")
    valid_data = processed_data.filter(
        lambda x: x['messages'] is not None and x['images'] is not None,
        num_proc=max_workers,
        desc="Filtering valid data"
    )

    print("Converting image column format...")
    valid_data = valid_data.cast_column('images', Sequence(feature=ImageData(decode=True)))

    print(f"Saving dataset to parquet format: {output_file}")
    valid_data.to_parquet(output_file)

    valid_count = len(valid_data)
    filtered_count = total_items - valid_count

    print("Conversion completed!")
    print(f"- Total input data: {total_items}")
    print(f"- Successfully converted: {valid_count}")
    print(f"- Filtered out: {filtered_count}")
    print(f"- Output file: {output_file}")

    thinking_count = sum(1 for item in valid_data if not item['messages'][0]['content'].endswith(' /no_think'))
    no_thinking_count = valid_count - thinking_count

    print("Mode distribution:")
    print(f"- thinking: {thinking_count}")
    print(f"- no_thinking: {no_thinking_count}")

    return valid_data

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Convert process.py generated data to multimodal conversation format")
    parser.add_argument("--input_files",
                       type=str,
                       nargs='+',
                       help="Input JSON file paths, can be multiple files")
    parser.add_argument("--input_pattern",
                       type=str,
                       help="Input file pattern, e.g.: 'results/generated_responses_*.json'")
    parser.add_argument("--output_file",
                       type=str,
                       required=True,
                       help="Output parquet file path")
    parser.add_argument("--max_workers",
                       type=int,
                       default=None,
                       help="Maximum concurrent processes, defaults to CPU count")
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()

    if not args.output_file.endswith('.parquet'):
        args.output_file += '.parquet'

    input_files = []

    if args.input_files:
        input_files = args.input_files
    elif args.input_pattern:
        import glob
        input_files = glob.glob(args.input_pattern)
        if not input_files:
            print(f"No files found matching pattern: {args.input_pattern}")
            return
    else:
        print("Error: Must specify --input_files or --input_pattern")
        return

    print(f"Found {len(input_files)} input files:")
    for f in input_files:
        print(f"  - {f}")

    convert_to_multimodal_format_with_map(
        input_files=input_files,
        output_file=args.output_file,
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    main()
