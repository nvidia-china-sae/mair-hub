# Distilling the Reasoning Ability of DeepSeek R1 into Qwen with the NeMo 2.0 Framework

DeepSeek R1 is an open-source large language model dedicated to solving logical reasoning tasks. It employs a Mixture of Experts (MoE) architecture and boasts 671B parameters. Through reinforcement learning, it has been trained to perform deep thinking (generating long-chain-of-thought), excelling in reasoning tasks and various specialized fields such as mathematics, programming, and scientific analysis.

Moreover, as per the [DeepSeek-R1](https://arxiv.org/abs/2501.12948) paper, the reasoning patterns of larger models can be distilled into smaller ones. Specifically, we can distill long-chain-of-thought (long-CoT) data that encapsulates reasoning processes from DeepSeek-R1 and directly fine-tune open-source models like Qwen and Llama. This simple distillation approach greatly enhances the reasoning capabilities of smaller models.

To illustrate the complete distillation process, we have prepared three notebooks demonstrating how to extract reasoning data from DeepSeek-R1 using the NIM API, how to train models with the distilled data, and how to evaluate the model.

* [1.generate_reasoning_data.ipynb](./1.generate_reasoning_data.ipynb) demonstrates the process of distilling reasoning data from DeepSeek-R1 using the NIM API.
* [2.qwen2_distill_nemo.ipynb](./2.qwen2_distill_nemo.ipynb) shows how to train open-source models with the distilled data. 
* [3.evaluation.ipynb](./3.evaluation.ipynb) shows how the evaluate the model.