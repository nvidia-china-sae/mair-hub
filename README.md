# MAIR-Hub

**MAIR-Hub** (MAIR stands for **M**ultimodal **AI** **R**esources.) is a central repository for **M**ultimodal **AI** **R**esources. This hub serves as a comprehensive collection of tutorials, code examples and other assets related to multimodal AI research and applications.

## Repository Structure

The following directories contain specialized resources for different aspects of multimodal AI:

| Directory | Description |
|-----------|-------------|
| [rl-tutorial](./rl-tutorial/) | Reinforcement Learning tutorials, including RL experiments with step-by-step guidance for reproduction |
| [speech-llm](./speech-llm/) | Speech LLM training recipes, including Qwen-omni-like speech2speech model training etc. |
| [external-resources](#external-resources) | Curated links to other valuable multimodal AI resources |

### RL-Tutorial

The [rl-tutorial](./rl-tutorial/) directory contains resources focused on reinforcement learning approaches in multimodal AI:

- [r1-zero](./rl-tutorial/r1-zero/): Tutorial of using the veRL framework to reproduce the reinforcement learning training process of DeepSeek-R1-Zero in the mathematics domain.
- [r1-like](./rl-tutorial/r1-like/): Tutorial of using the openRLHF framework to reproduce the reinforcement learning training process of DeepSeek-R1 in the mathematics domain.
- [vlm-R1](./rl-tutorial/vlm-R1/): Tutorial of using the veRL framework to train VLM models with reinforcement learning using both text and multimodal data to enhance reasoning capabilities in the mathematics domain.
- [cosyvoice_llm](./rl-tutorial/cosyvoice_llm/): A training recipe for using the veRL framework to conduct reinforcement learning experiments on the CosyVoice2 LLM in the speech-generation domain.
- [kdd_labs](./rl-tutorial/kdd_labs/): Hands-on labs for KDD 2025 Tutorial session, including distilling reasoning abilities from DeepSeek-R1 into smaller models using NeMo 2.0 Framework and GRPO training tutorial using NeMo RL tutorials.

### Speech-LLM

The [speech-llm](./speech-llm/) directory provides resources for training Speech LLMs:

- [qwen-omni-like](./speech-llm/qwen_omni_like/): Recipe for training **Qwen2.5-Omni-style** speech-to-speech (S2S) models.

### External Resources

This section provides links to valuable external tutorials and resources related to multimodal AI:

#### Reasoning and Knowledge Distillation

- [Distilling DeepSeek R1 into Qwen](https://github.com/NVIDIA/NeMo/blob/main/tutorials/llm/distill_deepseek_r1/REAMDE.rst): A tutorial demonstrating how to distill the reasoning abilities of DeepSeek R1 (a 671B parameter MoE model) into smaller models like Qwen using the NVIDIA NeMo 2.0 Framework. The repository includes notebooks for extracting reasoning data and training models with the distilled knowledge.

We are working on adding more tutorials and assets...

This project is licensed under the terms of the LICENSE file included in the repository.