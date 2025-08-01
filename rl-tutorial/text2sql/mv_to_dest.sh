#!/bin/bash

# 检查verl目录是否存在
if [ ! -d "verl" ]; then
    echo "错误: verl 目录不存在，请确保在正确的工作目录中运行此脚本"
    exit 1
fi

echo "开始移动文件..."

# 1. 移动工具类
echo "移动SQL工具文件..."
[ -f "sql_tool.py" ] && mv sql_tool.py verl/verl/tools/ && echo "✓ sql_tool.py 已移动"
[ -f "sql_execution_utils.py" ] && mv sql_execution_utils.py verl/verl/tools/utils/ && echo "✓ sql_execution_utils.py 已移动"

# 2. 移动sglang_rollout.py
echo "移动sglang_rollout.py..."
[ -f "sglang_rollout.py" ] && mv sglang_rollout.py verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py && echo "✓ sglang_rollout.py 已移动"

# 3. 创建 recipe/text2sql 目录并移动当前目录下所有文件
echo "创建 verl/recipe/text2sql 目录并移动当前目录下所有文件..."
mkdir -p verl/recipe/text2sql
mv * verl/recipe/text2sql
echo "✓ 当前目录下所有文件已移动到 verl/recipe/text2sql/"

echo "文件移动完成！"