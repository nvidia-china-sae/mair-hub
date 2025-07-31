# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

import argparse
import logging
import os
import tempfile
import re
import json
import random

import pandas as pd
import numpy as np


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# # Configuration constants
# DEFAULT_SYSTEM_CONTENT = (
#     "You are a data science expert. Your task is to understand database schemas and generate valid SQL queries "
#     "to answer natural language questions using SQLite database engine. You must conduct reasoning inside "
#     "<think> and </think> blocks every time you get new information. After reasoning, you need to explore "
#     "or verify database information, you can call a SQL execution tool by <tool_call> execute_sql </tool_call> "
#     "and it will return the query results between <tool_response> and </tool_response>. "
#     "You can execute SQL queries as many times as you want to explore the database structure and data. "
#     "When generating queries: (1) Only output information asked in the question, (2) Include all required "
#     "information without missing or extra data, (3) Think through query steps including analyzing questions, "
#     "summarizing findings, and verifying accuracy. If you find no further exploration is needed, you MUST "
#     "return your final SQL query enclosed within the <answer> </answer> tags."
# )

# Configuration constants
BEFORE_USER_CONTENT = (
    "You are a data science expert. Your task is to understand database schemas and generate valid SQL queries "
    "to answer natural language questions using SQLite database engine. You must conduct reasoning inside "
    "<think> and </think> blocks every time you get new information. After reasoning, you need to explore "
    "or verify database information, you can call a SQL execution tool by <tool_call> execute_sql </tool_call> "
    "and it will return the query results between <tool_response> and </tool_response>. "
    "You can execute SQL queries as many times as you want to explore the database structure and data. "
    "If you find no further exploration is needed, you MUST return your final SQL query enclosed within the <answer> </answer> tags."
)

AFTER_USER_CONTENT = (
    "Remember, you can call the SQL execution tool to explore and verify database information. When you have gathered sufficient information, you should only output the SQL query itself inside the <answer> </answer> tags, not the results produced by executing the query."
)


def extract_question_from_prompt(prompt):
    """
    从原始prompt中提取问题文本
    
    Args:
        prompt: 原始prompt列表或numpy数组，包含system和user消息
        
    Returns:
        str: 提取的问题文本
    """
    try:
        # 处理numpy数组
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()
        
        if not isinstance(prompt, list) or len(prompt) < 2:
            return ""
        
        # 获取最后一个消息（通常是user消息）
        user_message = prompt[-1]
        if not isinstance(user_message, dict) or 'content' not in user_message:
            return ""
        
        user_content = user_message.get("content", "")
        
        return user_content
        
    except Exception as e:
        logger.warning(f"提取问题失败: {e}")
        return ""


def process_single_row(row, current_split_name, row_index, db_root_path):
    """
    Process a single row of Text2SQL data for SearchR1-like format.

    Args:
        row: DataFrame row containing the original Text2SQL data
        current_split_name: Name of the current split (train/test)
        row_index: Index of the row in the DataFrame
        db_root_path: Root path for database files

    Returns:
        pd.Series: Processed row data in SearchR1 format
    """
    try:
        # 从原始prompt中提取问题
        original_prompt = row.get("prompt", [])
        question = extract_question_from_prompt(original_prompt)
        
        if not question:
            # 如果无法从prompt提取问题，直接返回None过滤掉该行
            return None
    except Exception as e:
        logger.error(f"第{row_index}行处理失败: {e}")
        return None

    # Build prompt structure for SearchR1 format
    user_content = question
    prompt = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": BEFORE_USER_CONTENT + "\n" + user_content + "\n" + AFTER_USER_CONTENT}]

    # Extract ground truth SQL from reward_model
    reward_model_data = row.get("reward_model")
    ground_truth_sql = ""
    if isinstance(reward_model_data, dict) and "ground_truth" in reward_model_data:
        ground_truth_sql = reward_model_data.get("ground_truth", "")

    # Process data source
    original_data_source = row.get("data_source", "")
    data_source_tagged = "sqlR1_" + str(original_data_source)

    # Get database ID and ability
    db_id = row.get("db_id", "")
    ability = row.get("ability", "sql-reasoning")

    # Build tools kwargs structure for SQL execution
    tools_kwargs = {
        "execute_sql": {
            "create_kwargs": {
                "ground_truth": ground_truth_sql,
                "question": question,
                "data_source": data_source_tagged,
                "db_id": db_id,
                "original_data": row.get("data", ""),
                "ability": ability,
                "db_root_path": db_root_path
            }
        }
    }

    # Build complete extra_info structure
    extra_info = {
        "index": row_index,
        "need_tools_kwargs": True,
        "question": question,
        "split": current_split_name,
        "tools_kwargs": tools_kwargs,
        "db_id": db_id,
        "original_prompt": original_prompt  # 保留原始prompt用于参考
    }

    # Handle metadata to avoid Parquet writing issues
    metadata = row.get("metadata")
    if metadata is None or (isinstance(metadata, dict) and len(metadata) == 0):
        metadata = None

    result = pd.Series(
        {
            "data_source": data_source_tagged,
            "prompt": prompt,
            "ability": ability,
            "reward_model": reward_model_data,
            "extra_info": extra_info,
            "metadata": metadata
        }
    )
    
    return result


def make_json_serializable(obj):
    """递归地使对象可JSON序列化"""
    try:
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (pd.Timestamp, pd.NaT.__class__)):
            return str(obj)
        elif isinstance(obj, dict):
            return {key: make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(item) for item in obj]
        else:
            # 先尝试检查是否为pandas的NaN/NaT
            try:
                if pd.isna(obj):
                    return None
            except (ValueError, TypeError):
                pass
            return str(obj)
    except Exception:
        return str(obj)


def save_random_sample_comparison(df_raw, df_processed, split_name):
    """保存随机样本的处理前后对比到sample_row.json文件"""
    try:
        # 随机选择一个索引
        random_idx = random.randint(0, min(len(df_raw), len(df_processed)) - 1)
        logger.info(f"选择第{random_idx}行作为样本进行对比保存")
        
        # 获取原始数据和处理后数据
        raw_sample = df_raw.iloc[random_idx]
        processed_sample = df_processed.iloc[random_idx]
        
        # 转换为可JSON序列化的格式
        raw_dict = {}
        for col in df_raw.columns:
            raw_dict[col] = make_json_serializable(raw_sample[col])
        
        processed_dict = {}
        for col in df_processed.columns:
            processed_dict[col] = make_json_serializable(processed_sample[col])
        
        # 构建对比数据
        comparison_data = {
            "sample_info": {
                "index": random_idx,
                "split": split_name,
                "timestamp": pd.Timestamp.now().isoformat()
            },
            "before_processing": raw_dict,
            "after_processing": processed_dict,
            "comparison_summary": {
                "original_columns": list(df_raw.columns),
                "processed_columns": list(df_processed.columns),
                "extracted_question": processed_dict.get("extra_info", {}).get("question", "未提取到问题") if "extra_info" in processed_dict else "未提取到问题"
            }
        }
        
        # 保存到sample_row.json文件
        sample_file_path = "sample_row.json"
        with open(sample_file_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已保存随机样本对比数据到: {sample_file_path}")
        
    except Exception as e:
        logger.error(f"保存随机样本对比数据失败: {e}")


def main():
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    processed_files = []

    # Process the text2sql dataset
    try:
        logger.info(f"Processing Text2SQL dataset from {args.input_file}")
        logger.info(f"Using database root path: {args.db_root_path}")
        
        # Load Text2SQL Parquet file
        df_raw = pd.read_parquet(args.input_file)
        logger.info(f"Loaded {len(df_raw)} rows from Text2SQL dataset")

        # 确定split名称
        if "train" in args.input_file.lower():
            split_name = "train"
        elif "test" in args.input_file.lower():
            split_name = "test"
        elif "val" in args.input_file.lower() or "dev" in args.input_file.lower():
            split_name = "validation"
        else:
            split_name = "processed"

        def apply_process_row(row):
            return process_single_row(row, current_split_name=split_name, row_index=row.name, db_root_path=args.db_root_path)

        # 处理数据并过滤掉None结果
        processed_results = df_raw.apply(apply_process_row, axis=1)
        
        # 检查是否有None值行
        null_mask = processed_results.isna().all(axis=1)
        valid_mask = ~null_mask
        
        # 过滤掉包含全None的行
        df_processed = processed_results[valid_mask]
        
        filtered_count = len(processed_results) - len(df_processed)
        if filtered_count > 0:
            logger.info(f"过滤掉 {filtered_count} 条没有有效question的数据，保留 {len(df_processed)} 条数据")
        else:
            logger.info(f"所有 {len(df_processed)} 条数据都有效")

        # 保存随机样本的处理前后对比
        if len(df_processed) > 0:
            save_random_sample_comparison(df_raw, df_processed, split_name)

        # Save processed DataFrame
        output_file_path = os.path.join(local_save_dir, f"{split_name}.parquet")
        df_processed.to_parquet(output_file_path, index=False)
        logger.info(f"Saved {len(df_processed)} processed rows to {output_file_path}")
        processed_files.append(output_file_path)

    except Exception as e:
        logger.error(f"Error processing Text2SQL dataset: {e}")
        return

    if not processed_files:
        logger.warning("No data was processed or saved")
        return

    logger.info(f"Successfully processed {len(processed_files)} files to {local_save_dir}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Text2SQL dataset to SearchR1 format and save to Parquet.")
    parser.add_argument("--input_file", default="/apps/data/SkyRL-SQL-653-data/train.parquet", 
                        help="Path to the input Text2SQL Parquet file.")
    parser.add_argument("--local_dir", default="~/data/sqlR1_processed", 
                        help="Local directory to save the processed Parquet files.")
    parser.add_argument("--hdfs_dir", default=None, 
                        help="Optional HDFS directory to copy the Parquet files to.")
    parser.add_argument("--db_root_path", default="/apps/data/OmniSQL-datasets/data/",
                        help="Root path for database files.")

    args = parser.parse_args()

    main() 