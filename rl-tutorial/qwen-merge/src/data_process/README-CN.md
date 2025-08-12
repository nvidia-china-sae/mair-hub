# 数据准备

本文档详细介绍了Qwen-Merge训练过程中三个阶段的数据准备流程。根据训练方法，我们将数据准备分为三个阶段：

1. **第一阶段 Projector Alignment**：使用图像-文本Caption数据对Projector结构进行预热
2. **第二阶段 ViT-LLM Alignment**：分两步进行多模态对齐，包括Caption & OCR任务和多样化任务
3. **第三阶段 Instruct Fine-Tuning**：通过蒸馏策略提升推理能力

## 第一阶段和第二阶段数据准备

### 数据集概述

**第一阶段数据集**：
- [LLaVA-ReCap-558K](https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-558K)

**第二阶段数据集**：
- 第二阶段第一步（Caption & OCR任务）：
  - LLaVA-ReCap-558K
  - [LLaVA-ReCap-CC3M](https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-CC3M)  
  - [LLaVA-ReCap-CC12M](https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-CC12M)
  - [synthdog-zh](https://huggingface.co/datasets/naver-clova-ix/synthdog-zh)
  - [synthdog-en](https://huggingface.co/datasets/naver-clova-ix/synthdog-en)
- 第二阶段第二步（多样化任务）：
  - [LLaVA-OneVision](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data)

### 数据下载和处理

运行以下命令下载并处理第一、二阶段训练数据：

```shell
# 处理LLaVA系列数据集（第一阶段和第二阶段-1）
python3 src/data_process/process_llava.py --output_dir ./LLaMA-Factory/data

# 处理SynthDOG数据集（第二阶段-1 OCR数据）
python3 src/data_process/process_synthdog.py --output_dir ./LLaMA-Factory/data

# 处理LLaVA-OneVision数据集（第二阶段-2 通用任务）
python3 src/data_process/process_onevision.py --output_dir ./LLaMA-Factory/data/llavaonevision_converted
```

## 第三阶段数据准备

第三阶段通过双重蒸馏策略提升模型推理能力，数据构建分为两个方面：

1. **文本推理数据转图像**：从Qwen3-32B蒸馏文本推理数据，将文本问题转化为图片形式
2. **多模态推理数据蒸馏**：从多模态推理模型蒸馏结果，规范多模态推理过程

### 文本推理数据处理

为了将Qwen3强大的文本推理能力扩展到视觉模态，我们采用文本推理数据转图像的策略。通过从Qwen3-32B蒸馏高质量的文本推理数据，并将文本问题转化为图片形式，让模型学会在视觉输入下进行类似的推理过程，从而提升多模态推理能力。

#### 步骤1：启动Qwen3-32B推理服务

首先启动Qwen3-32B的vLLM推理服务用于文本推理数据蒸馏：

```shell
VLLM_WORKER_MULTIPROC_METHOD=spawn python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --max-model-len 8192 \
    --served-model-name Qwen/Qwen3-32B
```

#### 步骤2：生成推理数据

使用Qwen3-32B对文本推理问题进行回答，主要基于以下数据集：
- [Chinese-DeepSeek-R1](https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k)
- [alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)

```shell
# 生成模型推理结果
python3 src/data_process/stage3/text2image/process.py \
    --model Qwen/Qwen3-32B \
    --api_ips localhost \
    --max_workers 32

# 基于原始答案评估和过滤低质量数据
python3 src/data_process/stage3/text2image/evaluate.py \
    --input_file generated_responses_Qwen3-32B.json \
    --model Qwen/Qwen3-32B \
    --api_ips localhost \
    --max_workers 32
```

#### 步骤3：文本转图像

安装必要的字体支持，并将文本问题转换为图像格式：

```shell
# 安装中文字体和其他必要字体
apt-get update && apt-get install -y \
    fonts-wqy-microhei \
    fonts-wqy-zenhei \
    fonts-arphic-uming \
    fonts-liberation \
    fonts-dejavu

# 将文本问题转换为图片格式，生成最终训练数据
python3 src/data_process/stage3/text2image/convert.py \
    --input_pattern ./evaluated_generated_responses_Qwen3-32B.json \
    --output_file ./LLaMA-Factory/data/text2image.parquet \
    --max_workers 32
```

### 多模态推理数据处理

多模态推理数据处理的目标是从现有的多模态推理模型中蒸馏出规范化的推理过程，让模型学会在多模态推理过程中思考范式。
用户可选择任意能力强的多模态推理模型，这里我们选择的是GLM-4.1V。
#### 步骤1：启动GLM-4.1V推理服务

**注意**：由于当前环境安装的vLLM版本可能不支持GLM-4.1V模型，建议在其他环境中基于vLLM最新版本启动服务。如需在当前环境运行，请先升级vLLM：

```shell
# 可选：若需在当前环境运行GLM-4.1V，先升级vLLM
# pip install --upgrade --upgrade-strategy only-if-needed vllm

# 启动GLM-4.1V推理服务
VLLM_WORKER_MULTIPROC_METHOD=spawn python3 -m vllm.entrypoints.openai.api_server \
    --model THUDM/GLM-4.1V-9B-Thinking \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --served-model-name THUDM/GLM-4.1V-9B-Thinking
```

#### 步骤2：生成多模态推理数据

从LLaVA-OneVision数据集中采样多模态数据，使用GLM-4.1V模型生成规范化的推理过程：

```shell
# 处理多模态数据，生成推理结果
python3 src/data_process/stage3/image/process.py \
    --model THUDM/GLM-4.1V-9B-Thinking \
    --api_ips localhost \
    --data_dir ./LLaMA-Factory/data/llavaonevision_converted \
    --max_workers 32 \
    --test_mode

# 评估和过滤低质量推理数据
python3 src/data_process/stage3/image/evaluate.py \
    --input_file glm_response.parquet \
    --output_file ./LLaMA-Factory/data/image_thinking.parquet \
    --judge_model THUDM/GLM-4.1V-9B-Thinking \
    --api_ips localhost \
    --max_workers 32
```

#### 步骤3：环境回滚（可选）

如果之前升级了vLLM版本，需要回滚到项目兼容版本：

```shell
# 回滚vLLM到项目兼容版本
cp -fr ./vllm/vllm/* /usr/local/lib/python3.12/dist-packages/vllm

# 回滚transformers到指定版本
pip install -U transformers==4.52.3 && \
cp -fr ./transformers/src/transformers/* /usr/local/lib/python3.12/dist-packages/transformers
```

## 数据处理完成检查

完成上述步骤后，请确认以下数据文件已正确生成：
```text
LLaMA-Factory/data/
├── LLaVA_ReCap_558K.parquet           # 第一阶段数据
├── LLaVA_ReCap_CC3M.parquet           # 第二阶段-1 Caption数据
├── LLaVA_ReCap_CC12M.parquet          # 第二阶段-1 Caption数据
├── synthdog_en.parquet                # 第二阶段-1 OCR数据
├── synthdog_zh.parquet                # 第二阶段-1 OCR数据
├── text2image.parquet                 # 第三阶段文本转图像数据
├── image_thinking.parquet             # 第三阶段多模态推理数据
└── llavaonevision_converted/          # 第二阶段-2 通用任务数据目录
    ├── ai2d_cauldron_llava_format.parquet
    └── ...
```

