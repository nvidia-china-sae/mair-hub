# Data Preparation

This document provides a detailed introduction to the data preparation process for the three stages of Qwen-Merge training. According to the training methodology, we divide data preparation into three stages:

1. **Stage 1: Projector Alignment**: Use image-text caption data to pre-train the Projector structure
2. **Stage 2: ViT-LLM Alignment**: Conduct multimodal alignment in two steps, including Caption & OCR tasks and diverse tasks
3. **Stage 3: Instruct Fine-Tuning**: Enhance reasoning capabilities through distillation strategy

## Stage 1 and Stage 2 Data Preparation

### Dataset Overview

**Stage 1 Dataset**:
- [LLaVA-ReCap-558K](https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-558K)

**Stage 2 Dataset**:
- Stage 2 Step 1 (Caption & OCR tasks):
  - LLaVA-ReCap-558K
  - [LLaVA-ReCap-CC3M](https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-CC3M)  
  - [LLaVA-ReCap-CC12M](https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-CC12M)
  - [synthdog-zh](https://huggingface.co/datasets/naver-clova-ix/synthdog-zh)
  - [synthdog-en](https://huggingface.co/datasets/naver-clova-ix/synthdog-en)
- Stage 2 Step 2 (Diverse tasks):
  - [LLaVA-OneVision](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data)

### Data Download and Processing

Run the following commands to download and process Stage 1 and Stage 2 training data:

```shell
# Process LLaVA series datasets (Stage 1 and Stage 2-1)
python3 src/data_process/process_llava.py --output_dir ./LLaMA-Factory/data

# Process SynthDOG dataset (Stage 2-1 OCR data)
python3 src/data_process/process_synthdog.py --output_dir ./LLaMA-Factory/data

# Process LLaVA-OneVision dataset (Stage 2-2 general tasks)
python3 src/data_process/process_onevision.py --output_dir ./LLaMA-Factory/data/llavaonevision_converted
```

## Stage 3 Data Preparation

Stage 3 enhances model reasoning capabilities through a dual-distillation strategy, with data construction conducted from two aspects:

1. **Text-to-Image Reasoning Data**: Distill text reasoning data from Qwen3-32B and convert text questions into image format
2. **Multimodal Reasoning Data Distillation**: Distill results from multimodal reasoning models to standardize the multimodal reasoning process

### Text Reasoning Data Processing

To extend Qwen3's powerful text reasoning capabilities to the visual modality, we adopt a text-to-image reasoning data strategy. By distilling high-quality text reasoning data from Qwen3-32B and converting text questions into image format, we enable the model to learn similar reasoning processes under visual input, thereby enhancing multimodal reasoning capabilities.

#### Step 1: Start Qwen3-32B Inference Service

First, start the vLLM inference service for Qwen3-32B for text reasoning data distillation:

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

#### Step 2: Generate Reasoning Data

Use Qwen3-32B to answer text reasoning questions, mainly based on the following datasets:
- [Chinese-DeepSeek-R1](https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k)
- [alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)

```shell
# Generate model inference results
python3 src/data_process/stage3/text2image/process.py \
    --model Qwen/Qwen3-32B \
    --api_ips localhost \
    --max_workers 32

# Evaluate and filter low-quality data based on original answers
python3 src/data_process/stage3/text2image/evaluate.py \
    --input_file generated_responses_Qwen3-32B.json \
    --model Qwen/Qwen3-32B \
    --api_ips localhost \
    --max_workers 32
```

#### Step 3: Text-to-Image Conversion

Install necessary font support and convert text questions to image format:

```shell
# Install Chinese fonts and other necessary fonts
apt-get update && apt-get install -y \
    fonts-wqy-microhei \
    fonts-wqy-zenhei \
    fonts-arphic-uming \
    fonts-liberation \
    fonts-dejavu

# Convert text questions to image format and generate final training data
python3 src/data_process/stage3/text2image/convert.py \
    --input_pattern ./evaluated_generated_responses_Qwen3-32B.json \
    --output_file ./LLaMA-Factory/data/text2image.parquet \
    --max_workers 32
```

### Multimodal Reasoning Data Processing

The goal of multimodal reasoning data processing is to distill standardized reasoning processes from existing multimodal reasoning models, enabling the model to learn thinking paradigms during multimodal reasoning processes.
Users can choose any powerful multimodal reasoning model; here we select GLM-4.1V.

#### Step 1: Start GLM-4.1V Inference Service

**Note**: Since the currently installed vLLM version may not support the GLM-4.1V model, it is recommended to start the service in another environment based on the latest vLLM version. If you need to run in the current environment, please upgrade vLLM first:

```shell
# Optional: If you need to run GLM-4.1V in the current environment, upgrade vLLM first
# pip install --upgrade --upgrade-strategy only-if-needed vllm

# Start GLM-4.1V inference service
VLLM_WORKER_MULTIPROC_METHOD=spawn python3 -m vllm.entrypoints.openai.api_server \
    --model THUDM/GLM-4.1V-9B-Thinking \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --served-model-name THUDM/GLM-4.1V-9B-Thinking
```

#### Step 2: Generate Multimodal Reasoning Data

Sample multimodal data from the LLaVA-OneVision dataset and use the GLM-4.1V model to generate standardized reasoning processes:

```shell
# Process multimodal data and generate reasoning results
python3 src/data_process/stage3/image/process.py \
    --model THUDM/GLM-4.1V-9B-Thinking \
    --api_ips localhost \
    --data_dir ./LLaMA-Factory/data/llavaonevision_converted \
    --max_workers 32 \
    --test_mode

# Evaluate and filter low-quality reasoning data
python3 src/data_process/stage3/image/evaluate.py \
    --input_file glm_response.parquet \
    --output_file ./LLaMA-Factory/data/image_thinking.parquet \
    --judge_model THUDM/GLM-4.1V-9B-Thinking \
    --api_ips localhost \
    --max_workers 32
```

#### Step 3: Environment Rollback (Optional)

If vLLM version was previously upgraded, you need to rollback to the project-compatible version:

```shell
# Rollback vLLM to project-compatible version
cp -fr ./vllm/vllm/* /usr/local/lib/python3.12/dist-packages/vllm

# Rollback transformers to specified version
pip install -U transformers==4.52.3 && \
cp -fr ./transformers/src/transformers/* /usr/local/lib/python3.12/dist-packages/transformers
```

## Data Processing Completion Check

After completing the above steps, please confirm that the following data files have been correctly generated:
```text
LLaMA-Factory/data/
├── LLaVA_ReCap_558K.parquet           # Stage 1 data
├── LLaVA_ReCap_CC3M.parquet           # Stage 2-1 Caption data
├── LLaVA_ReCap_CC12M.parquet          # Stage 2-1 Caption data
├── synthdog_en.parquet                # Stage 2-1 OCR data
├── synthdog_zh.parquet                # Stage 2-1 OCR data
├── text2image.parquet                 # Stage 3 text-to-image data
├── image_thinking.parquet             # Stage 3 multimodal reasoning data
└── llavaonevision_converted/          # Stage 2-2 general tasks data directory
    ├── ai2d_cauldron_llava_format.parquet
    └── ...
```
```