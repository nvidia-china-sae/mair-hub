# Qwen2.5-Omni-Like Speech-to-Speech Recipe

This repository contains scripts for training **Qwen2.5-Omni-style** speech-to-speech (S2S) models.

## 1. Model Architecture

An overview of the S2S model is presented below. For a detailed explanation, please refer to the original [Qwen2.5-Omni paper](https://github.com/QwenLM/Qwen2.5-Omni).

1. **Speech Understanding**  
   A frozen **[Whisper Large-v2](https://github.com/openai/whisper)** encoder extracts acoustic features from the input audio.
2. **Feature Projection (Speech Adapter)**  
   The *Speech Adapter* projects Whisper features into the embedding space of the **Thinker** LLM.
3. **Text Generation (Thinker LLM)**  
   The *Thinker* predicts text tokens from the projected features.
4. **Speech Synthesis (Talker LLM)**  
   The final hidden states, together with the embeddings of the generated text tokens, are fed into the **Talker** LLM to predict speech tokens.
5. **Speech Token Conversion**  
   A pre-trained **CosyVoice 2** token-to-wave module converts the speech tokens into waveforms.

<p align="center">
  <img src="assets/framework.png" width="800"/>
</p>

## 2. Training Stages

The training pipeline consists of two stages:

| Stage | Data | Trainable Modules | Description |
|-------|------|-------------------|-------------|
| **1** | LibriSpeech | Speech Adapter | Speech continuation + ASR |
| **2** | UltraChat + VoiceAssistant | Speech Adapter + LLM LoRA + Speech Codec LM (Talker) | CosyVoice 2 25 Hz tokens |

> **Note**  
> For the speech-continuation task, we use the prepared dataset available at [fixie-ai/librispeech_asr](https://huggingface.co/datasets/fixie-ai/librispeech_asr).  
> Additional details can be found in [this paper](https://arxiv.org/abs/2309.00916).

### 2.1 Prerequisites
Install the required Python packages:

```bash
# Base dependencies
pip install -r requirements.txt

# Extra packages needed for speech synthesis / demo
pip install -r requirements-cosyvoice.txt
```

### 2.2 Using `train.sh`

`train.sh` accepts two positional arguments, **stage** and **stop_stage**.  
All stages within `[stage, stop_stage]` (inclusive) are executed sequentially.

| Stage | Purpose |
|-------|---------|
| **0** | Download pretrained models and all datasets |
| **1** | Pre-train Speech Adapter on LibriSpeech (speech continuation + ASR) |
| **2** | SFT on UltraChat + VoiceAssistant with LoRA and speech output |

Example commands:

```bash
# Run stage-1 training
bash train.sh 1 1
```

## 3. Evaluation

### 3.1 Using `eval.sh`

`eval.sh` is organised into six stages:

| Stage | Action |
|-------|--------|
| **-1** | Average checkpoints (optional) |
| **0** | Clone CosyVoice repo and download auxiliary models |
| **1** | Launch a local **Gradio** S2S demo (`--share` to expose a public link) |
| **2** | Start inference servers for **VoiceBench** (only speech2text tasks) |
| **3** | Run VoiceBench clients to collect metrics |
| **4** | Distributed decoding with **CosyVoice-2** (speech2speech and speech2text tasks)|
| **5** | ASR decoding to compute WER of generated speech |

Typical usages:

```bash
# Quick demo (download + Gradio)
bash eval.sh 1 1

# VoiceBench evaluation (servers + clients)
bash eval.sh 2 2
bash eval.sh 3 3
# VoiceBench evaluation (offline decoding using distributed data sampler)
bash eval.sh 4 4
bash eval.sh 5 5
```

### 3.2 VoiceBench Scores
We mainly use [VoiceBench](https://github.com/MatthewCYM/VoiceBench) to assess speech-to-text quality.  

| Model | AlpacaEval | CommonEval | WildVoice | SD-QA | MMSU | OBQA | BBH | IFEval | AdvBench | Overall |
|-------|------------|------------|-----------|-------|------|------|-----|--------|----------|---------|
| Whisper-v3-turbo + Qwen 2.5 0.5B | 3.10 | 3.17 | 2.86 | 25.22 | 31.88 <br>(fail 6.64) | 36.04 <br>(5.49 % fail) | 52.4 | 31.10 / 28.63 | 94.81 | **50.18** |
| SFT Stage (UltraChat + VoiceAssistant) | 2.78 | 2.80 | 2.18 | 18.90 | 25.34 <br>(fail 16.8) | 24.12 <br>(11.4 % fail) | 50.3 | 13.95 | 93.65 | 42.38 |
| Pre-train (LibriSpeech speech-continuation + ASR) → SFT | 2.86 | 2.91 | 2.34 | 20.54 | 25.44 <br>(fail 9.86) | 25.44 <br>(fail 9.86) | 50.6 | 13.08 | 96.15 | 43.69 |
| Pre-train (LibriSpeech + GigaSpeech + PeopleSpeech) → SFT | 3.03 | 2.96 | 2.49 | 20.53 | 25.37 <br>(fail 40) | 24.62 <br>(24.17 % fail) | 48.1 | 12.47 | 96.15 | 44.09 |
| Pre-train (LibriSpeech + GigaSpeech + PeopleSpeech) → SFT + InstructS2S | 3.06 | 2.87 | 2.48 | 21.70 | 25.60 <br>(fail 8) | 25.27 <br>(5.2 % fail) | 50.9 | 14.85 | 94.81 | **44.59** |

### 3.3 Word Error Rate
For alignment between generated speech and text, we transcribe the speech with [parakeet-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) and compute the word error rate (WER).

| Model | WER (CommonEval subset) | Details |
|-------|------------------------|---------|
| Pre-train (LibriSpeech speech-continuation + ASR) → SFT | **7.96 %** | 131 insertions, 667 deletions, 292 substitutions |