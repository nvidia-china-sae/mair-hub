# Qwen-Merge

## Background

The release of large language models with powerful reasoning capabilities, such as DeepSeek R1 and Qwen3, has attracted widespread attention in recent years. Consequently, we began to focus on how to introduce reasoning abilities to VLMs. Through experiments, we found that directly training Qwen2.5-VL models with RLVR, while significantly improving model performance on downstream tasks, results in response styles similar to the original model with no significant increase in response length, making it difficult to stimulate reflection phenomena like R1 models (see figure below), and is more prone to hallucination. Another part of the work attempts to train multimodal reasoning models from scratch based on SFT and reinforcement learning (RLHF+RLVR). However, these works often rely on large amounts of high-quality closed-source data and complex training processes, and usually do not open-source training code, making them difficult to reproduce.

This work aims to provide a lightweight multimodal reasoning model training solution based entirely on open-source frameworks, models, and data, providing the community with a reproducible end-to-end training example.

<img src="./assets/response_comparison.png">

## Training Method

As shown in the figure below, we concatenate the vision Transformer (ViT) of Qwen2.5-VL with Qwen3 for alignment and distillation training. The advantage of this approach is that it can leverage the well-trained vision encoder, reduce the difficulty of text-visual modality alignment, and extend Qwen3's inherent text reasoning capabilities to the visual modality.

Training is divided into three stages:
- **Stage 1: Projector Alignment**: Freeze the vision encoder and LLM, use image-text pair data to pre-train the Projector structure
- **Stage 2: ViT-LLM Alignment**: Unfreeze the LLM and conduct multimodal alignment in two steps
- **Stage 3: Instruct Fine-Tuning**: Implement dual-distillation strategy to enhance reasoning capabilities

<img src="./assets/train_pipeline.png">

## Environment Setup

First prepare the Docker image, start the container, and pull the mair-hub project code:

```shell
# Pull image
docker pull vllm/vllm-openai:v0.9.1rc1

# Start container
docker run --gpus all -it --rm -v $(pwd):/workspace -p 8000:8000 --workdir /workspace vllm/vllm-openai:v0.9.1rc1 bash

# Pull mair-hub project
git clone https://github.com/nvidia-china-sae/mair-hub

# Switch to qwen-merge directory
cd mair-hub/rl-tutorial/qwen-merge
```

Since Qwen-Merge is a new model built based on Qwen2.5-VL and Qwen3, using this model for training and inference requires customized modifications to transformers, vllm, and LLaMA-Factory to support the new model architecture. The related modified code has been uploaded to the src directory. Run the following code for environment configuration:

```shell
# Pull open-source libraries and switch to specified versions
git clone https://github.com/hiyouga/LLaMA-Factory.git  && cd LLaMA-Factory && git checkout 83688b0b4d9483558cd69b23b5dca8bc7a1e11ae && pip install -e . && cd ..
git clone https://github.com/vllm-project/vllm.git  && cd vllm && git checkout ee5ad8d2c5f7126c344319da15526248f7b515d7 && cd ..
git clone https://github.com/huggingface/transformers.git && cd transformers && git checkout tags/v4.52.3 && cd ..

# Copy modified code to corresponding directories
cp -fr src/transformers/* transformers/ && cp -fr  ./transformers/src/transformers/* /usr/local/lib/python3.12/dist-packages/transformers
cp -fr src/vllm/* vllm/vllm && cp -fr ./vllm/vllm/* /usr/local/lib/python3.12/dist-packages/vllm
cp -fr src/LLaMA-Factory/* LLaMA-Factory/

# Install necessary dependencies
pip install qwen_vl_utils[decord]
pip3 install deepspeed
pip install wandb
```

## Model Construction

Run the following command to complete the construction of the Qwen-Merge model:

```shell
# Parameter description:
# --vlm-path: Source multimodal vision-language model path
# --llm-path: Target language model path  
# --materials-path: Related configuration file path
# --output-path: Output model save path
python3 src/model_merge/model_merger.py \
    --vlm-path "Qwen/Qwen2.5-VL-72B-Instruct" \
    --llm-path "Qwen/Qwen3-4B" \
    --materials-path "src/model_merge/materials" \
    --output-path "Qwen-Merge-VL-4B-base"
```

This command concatenates the vision Transformer (ViT) of Qwen2.5-VL with the language model of Qwen3 to construct a new multimodal reasoning model architecture. Through this approach, we can fully utilize the well-trained vision encoder and language model with strong reasoning capabilities.

## Data Preparation

All data used in this project comes from open-source datasets. Training is divided into three stages:

**Stage 1 and Stage 2 (Modal Alignment Stage)**:

- Stage 1 aligns the projector using image-text caption data
- Stage 2 aligns ViT-LLM in two steps for multimodal alignment

Stage 1 uses the [LLaVA-ReCap-558K](https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-558K) dataset. Stage 2 is divided into two steps: the first step uses [LLaVA-ReCap-CC3M](https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-CC3M) and [LLaVA-ReCap-CC12M](https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-CC12M) caption data, as well as [synthdog-en](https://huggingface.co/datasets/naver-clova-ix/synthdog-en) and [synthdog-zh](https://huggingface.co/datasets/naver-clova-ix/synthdog-zh) OCR data for modal alignment; the second step uses [LLaVA-OneVision](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data) diverse general data for model alignment.

**Stage 3 (Instruct Fine-Tuning Stage)**:

We enhance the model's reasoning capabilities through a dual-distillation strategy. Data construction is conducted from two aspects: on one hand, distilling text reasoning data from Qwen3-32B by converting text problems into image format; on the other hand, distilling results from multimodal reasoning models to standardize the multimodal reasoning process. The specific process is shown in the training method figure.

<img src="./assets/data_distillation.png">

The text data uses datasets such as [Chinese-DeepSeek-R1](https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k) and [alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca). Multimodal data is sampled from the LLaVA-OneVision dataset.

For detailed data preparation process, see [Data Preparation](./src/data_process/README.md)

## Training Steps

After completing data preparation, you can run the three training stages sequentially with the following commands:

```shell
# Stage 1: Projector Alignment
PYTHONPATH=./src llamafactory-cli train examples/train_qwen_merge/train_qwen_merge_vl_4B_stage1.yaml

# Stage 2.1: ViT-LLM Alignment (Caption & OCR)
PYTHONPATH=./src llamafactory-cli train examples/train_qwen_merge/train_qwen_merge_vl_4B_stage2.1.yaml

# Stage 2.2: ViT-LLM Alignment (Diverse Tasks)
PYTHONPATH=./src llamafactory-cli train examples/train_qwen_merge/train_qwen_merge_vl_4B_stage2.2.yaml

# Stage 3: Instruct Fine-Tuning (Dual Distillation)
PYTHONPATH=./src llamafactory-cli train examples/train_qwen_merge/train_qwen_merge_vl_4B_stage3.yaml
```

For multi-node training methods, see [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

## Inference and Evaluation

This project uses [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for evaluation, which also requires adaptation for Qwen-Merge. After completing the training environment setup, configure the evaluation environment with the following commands:

```shell
cd /workspace/mair-hub/rl-tutorial/qwen-merge
git clone https://github.com/open-compass/VLMEvalKit.git && cd VLMEvalKit && git checkout 21b00fa509c9028c33db5b6a3f1feda8f9e97645 && cd ..
cp -fr src/VLMEvalKit/* VLMEvalKit && cd VLMEvalKit && pip install -r requirements.txt
```

Before evaluation, you need to manually modify the model_path values for Qwen-Merge-VL-4B and Qwen-Merge-VL-8B in the `qwen2vl_series` section of the `vlmeval/vlm/config.py` file, specifying your trained model path:

```python
qwen2vl_series = {
    ...,
    "Qwen-Merge-VL-4B": partial(
        Qwen2VLChat,
        model_path="../LLaMA-Factory/outputs/Qwen-Merge-VL-4B-stage3",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_vllm=True,
        post_process=True,
        max_new_tokens=8192
    ),
    "Qwen-Merge-VL-8B": partial(
        Qwen2VLChat,
        model_path="../LLaMA-Factory/outputs/Qwen-Merge-VL-8B-stage3",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_vllm=True,
        post_process=True,
        max_new_tokens=8192
    )
}
```

Run evaluation:

```shell
# Set OpenAI API (for GPT evaluation)
export OPENAI_API_KEY=your_openai_api_key
export OPENAI_API_BASE=https://api.openai.com/v1

# Specify evaluation task and model
TASK_NAME=MathVista_MINI
MODEL_NAME=Qwen-Merge-VL-4B
python3 run.py --data ${TASK_NAME} --model ${MODEL_NAME} --verbose --use-vllm --reuse --judge chatgpt-0125
```

## Results

We conducted comprehensive testing on multiple standard multimodal reasoning evaluation tasks, with results shown in the table below:

| Datasets | Qwen2.5-VL-3B-Instruct | Qwen2.5-VL-7B-Instruct | Qwen-Merge-VL-4B-stage3 | Qwen-Merge-VL-8B-stage3 |
|----------|------------------------|------------------------|-------------------------|-------------------------|
| MMStar | 56.40 | 64.27 | 60.46 | 61.86 |
| MMMU_DEV_VAL | 49.33 | 54.00 | 54.67 | 58.00 |
| LogicVista | 39.37 | 47.43 | 47.20 | 48.77 |
| MathVista_MINI | 61.70 | 66.50 | 66.40 | 67.90 |
| MathVision_MINI | 21.51 | 27.30 | 28.95 | 34.21 |
| ScienceQA_VAL | 79.40 | 89.46 | 91.13 | 91.75 |
| VisOnlyQA | 38.76 | 43.34 | 43.27 | 44.18 |
| WorldMed-QA-V | 29.59 | 25.70 | 29.40 | 28.00 |
| TextVQA | 79.32 | 85.36 | 68.56 | 69.97 |
| OCRBench | 82.60 | 88.20 | 74.80 | 75.50 |
| MMVet | 35.32 | 37.75 | 25.87 | 33.67 |
| AI2D_TEST | 78.14 | 81.06 | 76.13 | 78.36 |
| ChartQA_TEST | 83.88 | 85.96 | 65.56 | 66.00 |

### Performance Analysis

Experimental results show that Qwen-Merge-VL demonstrates significant improvements compared to Qwen2.5-VL on many mathematical and logical reasoning tasks, particularly with Qwen-Merge-VL-4B performing even better than the larger Qwen2.5-VL-7B on multiple tasks. We believe this is mainly attributed to the reasoning capabilities of the underlying language model Qwen3 being generalized to the visual modality.

However, we also observed that Qwen-Merge-VL performs significantly lower than Qwen2.5-VL on certain tasks (such as TextVQA, OCRBench, etc.). We believe this is mainly influenced by two factors:

1. **Data Quality Issues**: The open-source datasets we used have not undergone refined quality screening and ratio adjustment, resulting in insufficient training data for certain specific tasks
2. **Training Technology Limitations**: The current training process only includes alignment and instruct fine-tuning stages, while mainstream methods typically also introduce advanced training techniques such as rejection sampling, RLHF, and RLVR, which play important roles in further improving model performance

It should be emphasized that our goal is not to build a general multimodal model, but to provide a feasible training methodology, demonstrating how to build Vision-Language Models with reasoning capabilities at relatively low cost. Users can reference our complete process and choose other large language models and vision encoders to customize models that meet their own needs.

## Next Steps

1. Introduce rejection sampling, RLHF+RLVR and other methods to further improve model performance
2. Provide training recipes optimized for vertical domain scenarios
