# CosyVoice2 LLM GRPO Tutorial

This recipe demonstrates how to train the **CosyVoice2** LLM with the **GRPO** algorithm using the [veRL](https://github.com/volcengine/verl) framework.

* **Input**: Chinese text
* **Output**: CosyVoice2 speech tokens

## Table of Contents
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Reward Function & ASR Server](#reward-function--asr-server)
- [GRPO Training](#grpo-training)
- [Model Merge & Evaluation](#model-merge--evaluation)
- [Single-Utterance Inference](#single-utterance-inference)

## Environment Setup

Stage `-1` of `run.sh` installs every dependency:

```bash
bash run.sh -1 -1   # only run stage -1
```

It performs:
1. Clone and install **vllm** (without Megatron).
2. Clone **CosyVoice** source code to `/workspace/CosyVoice` and install `requirements-cosyvoice.txt`.
3. Download the TTS codec model `iic/CosyVoice2-0.5B` via **ModelScope** to `/workspace/CosyVoice2-0.5B`.

After the script finishes you should have:
* working `vllm` installation
* CosyVoice Python package available in `$PYTHONPATH`
* `/workspace/CosyVoice2-0.5B` directory with the TTS codec checkpoints

## Data Preparation

Stage `0` converts raw JSONL files to the parquet format expected by veRL:

```bash
bash run.sh 0 0
```

`prepare_data.py` expects a JSON/JSONL file with the following minimal schema:

```jsonc
{
  "text": "一句待合成的中文句子"
}
```

It produces two parquet files:

```
data/parquet_tiny/train.parquet
data/parquet_tiny/test.parquet
```

Each sample is automatically wrapped into a chat-style prompt with a special system token `<|SPEECH_GENERATION_START|>` so that the LLM learns to output CosyVoice speech tokens.

## Reward Function & ASR Server

To compute rewards we run a lightweight server that:

1. Converts generated speech tokens back to a 16 kHz waveform with **CosyVoice2**.
2. Transcribes it with **SenseVoice** ASR.
3. Computes the pinyin-level WER w.r.t ground-truth text and converts it to a score in \[0 .. 1\].

Start the server (stage `1`) in a dedicated terminal / GPU:

```bash
bash run.sh 1 1
# Triton server listens on ports 8000/8001/8002
```

The Python implementation lives in [`reward_tts.py`](./reward_tts.py).

## GRPO Training

Run stage `2` to start GRPO training:

```bash
bash run.sh 2 2
```

Important CLI arguments used in the call to `verl.trainer.main_ppo`:

* `algorithm.adv_estimator=grpo` – switch from PPO to GRPO
* `data.train_files=data/parquet_aishell3/train.parquet` & `data.val_files=data/parquet_aishell3/test.parquet`
* `actor_rollout_ref.model.path=/workspace/rl/llasa_cosyvoice2_token_qwen_0.5b/checkpoint-885000` – pretrained CosyVoice2 LLM
* `custom_reward_function.path=reward_tts.py` – reward function described above
* `trainer.total_epochs=1` – train for one epoch (adjust as needed)

Tune `CUDA_VISIBLE_DEVICES`, batch sizes and learning rate according to your hardware.

## Model Merge & Evaluation

After training completes we gather the sharded FSDP weights and dump a HuggingFace-style checkpoint (stage `3`):

```bash
bash run.sh 3 3   # merges weights into $llm_path/merged_hf_model
```

We can then evaluate the model on the zero-shot Chinese test set (stage `4`):

```bash
bash run.sh 4 4
```

This launches distributed inference via `infer_dist.py` and computes WER with `scripts/compute_wer.sh`.

## Single-Utterance Inference

For a quick demo run stage `5`:

```bash
bash run.sh 5 5
```

It synthesises a tongue-twister using the merged checkpoint and prints the path to the generated audio.

---

Happy TTS fine-tuning! :musical_note: 