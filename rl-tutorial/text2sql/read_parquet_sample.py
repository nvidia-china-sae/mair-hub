#!/usr/bin/env python3
import pandas as pd
import json
import sys
import os
import numpy as np

def read_parquet_and_save_sample(parquet_path, output_file="sample_row_sql.json"):
    """
    读取parquet文件并保存一行数据到文件中
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(parquet_path):
            print(f"错误：文件 {parquet_path} 不存在")
            return False
        
        # 读取parquet文件
        print(f"正在读取文件: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        
        print(f"文件读取成功！")
        print(f"数据行数: {len(df)}")
        print(f"数据列数: {len(df.columns)}")
        print(f"列名: {list(df.columns)}")
        
        if len(df) == 0:
            print("警告：文件中没有数据行")
            return False
        
        # 随机获取一行数据
        random_index = np.random.randint(0, len(df))
        random_row = df.iloc[random_index]
        
        # 将随机行转换为字典
        row_dict = random_row.to_dict()
        
        # 处理可能的非JSON序列化类型
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
        
        # 应用序列化处理
        for key, value in row_dict.items():
            row_dict[key] = make_json_serializable(value)
        
        # 保存到JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(row_dict, f, ensure_ascii=False, indent=2)
        
        print(f"第一行数据已保存到: {output_file}")
        
        # 打印第一行数据的概览
        print("\n第一行数据预览:")
        for key, value in row_dict.items():
            # 限制显示长度
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"错误：{e}")
        return False

if __name__ == "__main__":
    # parquet_file = "/apps/data/sql_r1/train.parquet"
    parquet_file = "/apps/data/SkyRL-SQL-653-data/train.parquet"
    success = read_parquet_and_save_sample(parquet_file)
    
    if success:
        print("\n任务完成成功！")
    else:
        print("\n任务执行失败！")
        sys.exit(1) 