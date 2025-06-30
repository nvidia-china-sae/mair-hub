# R1-Zero RL Tutorial

In this tutorial, we will use the veRL framework to reproduce the reinforcement learning training process of DeepSeek-R1-Zero in the mathematics domain. The content includes:

- [R1-Zero RL Tutorial](#r1-zero-rl-tutorial)
  - [veRL Introduction](#verl-introduction)
    - [Environment Setup](#environment-setup)
    - [Key Parameters](#key-parameters)
  - [Data Preparation](#data-preparation)
    - [Data Format](#data-format)
    - [Processing Data](#processing-data)
    - [Dataset Description](#dataset-description)
  - [Custom Reward Function and Prompt Template](#custom-reward-function-and-prompt-template)
    - [Reward Function](#reward-function)
    - [Prompt Template](#prompt-template)
  - [Starting Training](#starting-training)
  - [Experiment Curves](#experiment-curves)
  - [Evaluation Experiments](#evaluation-experiments)
    - [Evaluation Setup](#evaluation-setup)
    - [Evaluation Results](#evaluation-results)
    - [Reflection Behavior Analysis](#reflection-behavior-analysis)
    - [Case Study](#case-study)

## veRL Introduction

This section provides a brief introduction to veRL installation and key parameters. If you're already familiar with veRL, you can skip directly to the [Data Preparation](#data-preparation) section.

### Environment Setup

Use the official veRL Docker image with the following command to start a container:

```bash
docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN \
 -v </path/need/to/mount>:</path/in/docker> \
 whatcanyousee/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te2.0-megatron0.11.0-v0.0.6

# veRL now supports vllm 0.8+ versions, you can use the following image
docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN \
 -v </path/need/to/mount>:</path/in/docker> \
 hiyouga/verl:ngc-th2.6.0-cu120-vllm0.8.2
```

After logging into the container, install verl and necessary dependencies:

```bash
# install the nightly version (recommended)
git clone https://github.com/volcengine/verl && cd verl && pip3 install -e .

# Since our experiments are on math tasks, we also need to install math-verify as a reward function
pip install math-verify[antlr4_9_3]
```

> [!TIP]
> 1. If you need to use wandb to monitor experiments, you need to login with `wandb login`
> 2. If you're using HuggingFace datasets that require authentication, you need to login with `huggingface-cli login`

### Key Parameters

| Category | Parameter | Description |
| --- | --- | --- |
| Basic Parameters | actor_rollout_ref.model.path | Original model parameter path |
| Basic Parameters | data.train_files<br>data.val_files | Training and validation data paths. Single path like data/orz/train.parquet, also supports multiple paths like "[data/orz/train.parquet,data/gsm/train.parquet]" |
| Basic Parameters | data.custom_cls.path | The path to the file containing your customized dataset class |
| Basic Parameters | data.custom_cls.name | The name of the dataset class within the specified file |
| Basic Parameters | custom_reward_function.path | The path to the file containing your customized reward function |
| Basic Parameters | custom_reward_function.name | The name of the reward function within the specified file. Default is 'compute_score' |
| Training Parameters | trainer.total_epochs | Number of epochs to train on the entire training set |
| Training Parameters | data.train_batch_size | Number of prompts consumed in each RL step |
| Training Parameters | actor_rollout_ref.rollout.n | How many responses to generate for each prompt during rollout. Must be greater than 1 for GRPO and GLOO |
| Training Parameters | actor_rollout_ref.rollout.temperature | Temperature coefficient for generating responses during rollout. Higher values result in stronger randomness |
| Training Parameters | actor_rollout_ref.actor.ppo_mini_batch_size | Batch size when updating actor parameters after rollout, must be less than and divisible by `data.train_batch_size` (e.g. `data.train_batch_size`=16, `actor_rollout_ref.actor.ppo_mini_batch_size`=8 or 16) |
| Training Parameters | algorithm.adv_estimator | RL algorithm used, currently supports PPO, GRPO, etc. |
| Training Parameters | algorithm.kl_ctrl.kl_coef | KL coefficient when calculating token-level rewards. Should be set to 0 for original GRPO |
| GRPO Parameters | actor_rollout_ref.actor.use_kl_loss | Whether to use external KL loss. Should be set to True for original GRPO algorithm |
| GRPO Parameters | actor_rollout_ref.actor.kl_loss_coef | Coefficient for external KL loss |
| vllm Parameters | actor_rollout_ref.rollout.gpu_memory_utilization | The proportion of the remaining GPU memory allocated for kv cache after other models have initialized when using vllm |
| vllm Parameters | actor_rollout_ref.rollout.tensor_model_parallel_size | TP size during rollout, only effective for vllm |

Other common training parameters, such as lr and warmup, are not detailed here. For specific settings, refer to chapters 2 and 3.

## Data Preparation

### Data Format

veRL only supports parquet format files by default, and they must conform to certain format requirements.

The default format for each sample in the RLHF stage in veRL is as follows:

```json
{
    "data_source": "pe-nlp/math-cl",
    "prompt": [
        {
            "role": "user",
            "content": question
        }
    ],
    "ability": "math",
    "reward_model": {
        "style": "rule",
        "ground_truth": solution
    },
    "extra_info": {
        'split': "train",
        'index': idx
    }
}
```

* data_source: Can be set to the name of the sample's dataset
* prompt: A list of chat messages. During training, the framework will automatically call tokenizer's apply_chat_template to process and tokenize
* ability: Task classification
* reward_model: For RL processes like R1, we set it as a rule-based reward function and pass the problem answer in ground_truth
* extra_info: Not currently used by the framework

### Processing Data

We need to write processing scripts for datasets according to the format above. Specific examples can be found in veRL's official [GSM8K processing script](https://github.com/volcengine/verl/blob/main/examples/data_preprocess/gsm8k.py) and the recipe's processing script [train_dataset.py](./train_dataset.py).

We need to run the processing scripts before training to convert the data to parquet format:

```bash
# Training set
python train_dataset.py

# Validation sets
python test_aime24.py
python test_math500.py
```

### Dataset Description

The training set for this tutorial comes from the MATH dataset and DeepScaler dataset. For both datasets, we filtered out problems with unverifiable answers using math-verify.

- MATH dataset: Selected ~8.4k samples from Level 3-5, with relatively low difficulty
- DeepScaler dataset:
    - Randomly sampled 16k samples
    - Generated 8 responses for each prompt using Qwen2.5-7B-Instruct and calculated pass rates
    - Filtered out samples with failed generation and pass rate of 1, leaving `12.7k` samples
    - Divided data into simple (pass rate > 65%), medium (pass rate 15%-65%), and difficult (pass rate < 15%) categories

- Curriculum learning:
    - First trained on the simple 8K data for 2 episodes, then trained on simple, medium, and difficult subsets for 1, 15, and 20 episodes respectively
    - Copied datasets according to preset episodes and concatenated them in sequence to create the final training set [pe-nlp/math-cl](https://huggingface.co/datasets/pe-nlp/math-cl)

## Custom Reward Function and Prompt Template

### Reward Function

The latest veRL now supports defining reward functions by passing in Python files. In this experiment, we use [math_reward.py](./math_reward.py) as the reward function.

Its main logic is:
- Check if the model output follows the format `<think>...</think>\s*<answer>...</answer>`. If not, return 0
- Extract the answer wrapped in `boxed` from `<answer>...</answer>`. If it doesn't exist, return 0
- Use math-verify to check if the extracted answer matches the ground truth answer. If they match, return 1; otherwise, return 0

### Prompt Template

Referencing the Open-Reasoner-Zero paper, we use the following prompt template:

```python
(
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
    "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. "
    "User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. "
    "And your final answer will be extracted automatically by the \\boxed{{}} tag. {prompt}\n"
    "Assistant: <think>"
)
```

The customized prompt template and processing logic is implemented in [r1_dataset.py](./r1_dataset.py) which is passed to veRL during training.

## Starting Training

The training script is available in [run_qwen2.5-7b.sh](./run_qwen2.5-7b.sh). You can refer to this script for training.

```bash
bash run_qwen2.5-7b.sh
```

## Experiment Curves

![Response Length Curve](./assets/response.png)

![Reward Curve](./assets/reward.png)
 
- The model nearly converged after only 100 steps on the MATH dataset, with response length stabilizing. However, when switching to the medium difficulty subset of the second dataset, response length suddenly increased while reward dropped sharply
- As the model continued training on the medium and difficult subsets, response length peaked at nearly 1100 tokens before converging at around 1000 tokens
- On both datasets, in the early stages, there was a phenomenon of response length and reward increasing simultaneously (consistent with the R1 paper). However, in the later stages, response length stabilized while reward continued to increase

## Evaluation Experiments

### Evaluation Setup

We selected 3 benchmark datasets widely used for evaluating reasoning models, including AIME 2024 and MATH 500 (both mathematics datasets), and GPQA-Diamond (a PhD-level dataset covering physics, chemistry, biology, and other fields).

For each prompt, we generated 8 responses (temperature=0.6) and calculated the average accuracy, or Pass@1[8]. The specific evaluation code can be found in [R1-Evaluation](https://github.com/zpqiu/R1-Evaluation).

### Evaluation Results

|  | AIME 2024 | MATH 500 | GPQA-Diamond |
| --- | --- | --- | --- |
| Qwen2.5-7B | 4.58 | 59 | 30.38 |
| Qwen2.5-7B-Instruct | 12.08 | 77.12 | 41.5 |
| **Qwen2.5-7B-GRPO** | 17.92 | 78.75 | 38.37 |

- After RL training, there was a significant improvement compared to the Base model, validating the effectiveness of the RL training paradigm based on Base models relying solely on outcome and rule-based reward functions
- The RL-trained model also showed improvement on the OOD GPQA dataset (non-mathematical problems, mainly chemistry and physics), indicating that the activated reasoning ability has generalization capability
- The RL model outperformed the 7B Instruct model on both mathematical benchmark datasets, with a particularly notable improvement on the more difficult AIME dataset

### Reflection Behavior Analysis

We tallied the percentage of responses containing reflection tokens ("reevaluate", "reevaluation", "recheck", "rethink", "check again", "re-evaluate", "try again", "re-examine") generated by different models across the 3 test sets:

| Reflection Token Occurrence | AIME 2024 | MATH 500 | GPQA-Diamond |
| --- | --- | --- | --- |
| Qwen2.5-7B | 4.16% | 0.6% | 2% |
| Qwen2.5-7B-Instruct | 5.41% | 0.75% | 1.25% |
| **Qwen2.5-7B-GRPO** | 11.67% | 4.5% | 9.50% |

We can observe:

- The original Base and Instruct models also exhibit a certain proportion of reflection behaviors, and do so more frequently on the most difficult AIME dataset
- After RL training, compared to the Base model, the RL model significantly increased reflection behaviors across all 3 datasets, with a more notable improvement on the OOD GPQA dataset

### Case Study

Below is a case where the entire thinking process includes step-by-step reasoning and reflection behaviors:

<details>
<summary>Click to view</summary>

```latex
Question: If \\[\\sqrt[3]{2} = a + \\cfrac{1}{b + \\cfrac{1}{c + \\cfrac{1}{d + \\dotsb}}},\\] where $a,$ $b,$ $c,$ $d$ are positive integers, compute $b.$

<think>
To find the value of \( b \) in the continued fraction representation of \( \sqrt[3]{2} \), we start with the equation:

\[
\sqrt[3]{2} = a + \cfrac{1}{b + \cfrac{1}{c + \cfrac{1}{d + \dotsb}}}
\]

First, we need to determine the integer part \( a \) of \( \sqrt[3]{2} \). Since \( 1^3 = 1 \) and \( 2^3 = 8 \), it follows that \( 1 < \sqrt[3]{2} < 2 \). Therefore, \( a = 1 \). This allows us to write:

\[
\sqrt[3]{2} = 1 + \cfrac{1}{b + \cfrac{1}{c + \cfrac{1}{d + \dotsb}}}
\]

Next, we isolate the fractional part:

\[
\sqrt[3]{2} - 1 = \cfrac{1}{b + \cfrac{1}{c + \cfrac{1}{d + \dotsb}}}
\]

We need to find the value of \( b \). To do this, we approximate \( \sqrt[3]{2} - 1 \):

\[
\sqrt[3]{2} - 1 \approx 1.25992 - 1 = 0.25992
\]

We take the reciprocal of \( 0.25992 \):

\[
\frac{1}{0.25992} \approx 3.859
\]

Since \( b \) must be an integer, we take \( b = 3 \). This gives us:

\[
\sqrt[3]{2} - 1 \approx \frac{1}{3 + \cfrac{1}{c + \cfrac{1}{d + \dotsb}}}
\]

We now isolate the new fractional part:

\[
\sqrt[3]{2} - 1 - \frac{1}{3} = \frac{1}{c + \cfrac{1}{d + \cfrac{1}{e + \dotsb}}}
\]

Simplifying the left side:

\[
\sqrt[3]{2} - 1 - \frac{1}{3} = 0.25992 - 0.33333 = -0.07341
\]

Since we made an approximation, we need to recheck with more precision. However, the continued fraction approach suggests we should re-evaluate the integer part more accurately. Given the continued fraction representation, we can use the known result for \( \sqrt[3]{2} \):

\[
\sqrt[3]{2} = 1 + \cfrac{1}{3 + \cfrac{1}{3 + \cfrac{1}{3 + \cfrac{1}{3 + \dotsb}}}}
\]

This confirms that \( b = 3 \). Therefore, the value of \( b \) is:

\[
\boxed{3}
\]

</think> <answer> \boxed{3} </answer>
```
</details> 