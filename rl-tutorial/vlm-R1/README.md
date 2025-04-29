# VLM Reinforcement Learning (RL) Tutorial

This tutorial will introduce how to use the veRL framework, combined with multimodal data in the mathematics domain, to train VLM models through reinforcement learning, thereby enhancing their reasoning abilities. The main contents include:

- [Environment Installation](#environment-installation)
- [Data Preparation](#data-preparation)
- [Custom Reward Function and Prompt Template](#custom-reward-function-and-prompt-template)
- [Training](#training)
- [Experiment Curves](#experiment-curves)
- [Evaluation Experiments](#evaluation-experiments)

## Environment Installation

This section will briefly introduce the installation method and key parameters of veRL. If you are already familiar with veRL, you can skip directly to the [Data Preparation](#data-preparation) section.

### Environment Preparation

Use the official veRL image and start the container with the following command:

```bash
# veRL now supports vllm 0.8+ version, you can use the following image to start
docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN \
 -v </path/need/to/mount>:</path/in/docker> \
 hiyouga/verl:ngc-th2.6.0-cu120-vllm0.8.2
```

After logging into the container, install verl and necessary dependencies

```bash
# install the nightly version (recommended)
git clone https://github.com/volcengine/verl && cd verl && pip3 install -e .

# Our subsequent experiments are conducted on mathematical tasks, so we need to install math-verify as a reward function
pip install math-verify[antlr4_9_3]

# If you need to use wandb to monitor experiments, you need to login with wandb
wandb login

# If you are using HuggingFace datasets that require authentication, you need to login to your hf account with huggingface-cli
huggingface-cli login
```


## Data Preparation

### Data Format

veRL currently only supports data files in parquet format, and the data needs to meet specific structural requirements.

In the RL phase of veRL, each sample should follow the data format below:

```json
{
    "data_source": "hiyouga/geometry3k",
    "prompt": [{
        "content": "<image>Find x. Round to the nearest tenth, if necessary. Let's think step by step and output the final answer within \\boxed{}.", 
        "role": "user"
    }],
    "images": [{"bytes": <class 'bytes'>}],
    "ability": "math",
    "reward_model": {
        "style": "rule",
        "ground_truth": "11"
    },
    "extra_info": {
        "answer": "11",
        "split": "train"
    }
}
```

* data_source: Identifies the dataset source of the sample, facilitating data tracing and management.
* prompt: Chat messages stored in list form. During training, the framework will automatically call the tokenizer's apply_chat_template method to process and tokenize them.
* images: Image data stored in list form, with data type bytes. The number of images should strictly correspond to the number of `<image>` tags in the prompt.
* ability: Describes the task type or ability domain involved in the sample.
* reward_model: In the R1 stage of RL training, it is recommended to use a rule-based reward function and fill in the standard answer to the question in the answer field.
* extra_info: This field is not currently directly used by the framework and can be left empty or customized according to actual needs.

### Data Processing

We need to write corresponding processing scripts for datasets according to the above data format. Specific implementation can refer to the [geometry3k processing script](https://github.com/volcengine/verl/blob/main/examples/data_preprocess/geo3k.py) provided by veRL.

### Dataset Introduction

The training set used in this tutorial is integrated from multiple open-source datasets, including [geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k), [MathVision](https://huggingface.co/datasets/MathLLMs/MathVision), [polymath](https://huggingface.co/datasets/him1411/polymath), [SceMQA-main](https://huggingface.co/datasets/Haozy/SceMQA-main), and [We-Math](https://huggingface.co/datasets/We-Math/We-Math). A total of approximately 11K samples, covering various mathematical problems involving images, text, and image-text combinations.

## Custom Reward Function and Prompt Template

### Reward Function

The latest version of veRL supports customizing Reward functions by passing in Python files. In this experiment, we use [math_verify_for_dapo.py](./math_verify_for_dapo.py) as the reward function.

Its core logic is as follows:
- Extract the answer wrapped in `boxed` from the model output, and return -1 if not detected;
- Use the math-verify tool to compare the extracted answer with the ground truth, return 1 if consistent, otherwise return 0.

### Prompt Template

Multimodal data has concatenated prompts with questions for each sample, with prompts as shown below:

```python
Let's think step by step and output the final answer within \\boxed{}.
```
The chat template defaults to using the model Tokenizer's Chat Template.



## Training

### Starting RAY

For single-node training, the command to start ray is as follows:
```bash
ray start --head --node-ip-address 0.0.0.0 --port=6379 --block 
```

For multi-node training, the command to start ray is as follows:
```bash
# Assuming there are 4 nodes
# Execute on the master node
ray start --head --node-ip-address=${MASTER_NODE_IP} --port=6379 --block 

# Execute on other nodes
ray start --address ${MASTER_NODE_IP}:6379 --block 
```

### Starting Training

Our experiments are based on the Qwen2.5-VL-7B-Instruct model, running on a single machine with 8 GPUs. After training with the multimodal mathematical data introduced in this article, the model's performance has significantly improved compared to the baseline. Run the following command on the master node to start training:

```bash
# Train with multimodal data
bash run_qwen2.5-vl-7b-instrcut-multimodal.sh
```

## Experiment Curves
The figure below shows the trend of reward values during the training process. As training progresses, the reward value steadily increases, indicating that the model's ability in mathematical reasoning tasks is continuously strengthening:

![reward curve](./assets/reward.png)
The figure below shows the trend of model response length during the training process. It can be observed that as training deepens, the model's response length increases somewhat, but the growth is moderate, typically stabilizing at around 400 tokens, avoiding excessively lengthy outputs:
![response length curve](./assets/response-length.png)




## Evaluation Experiments

### Text Task Evaluation

For text task evaluation, we selected 3 benchmark datasets widely used for evaluating reasoning models:
- **AIME 2024**: American Mathematics Invitational Examination problem set, with high difficulty
- **MATH 500**: A comprehensive test set containing various mathematical problems
- **GPQA-Diamond**: Doctoral-level questions in physics, chemistry, and biology

The evaluation method adopts Pass@1[8], which generates 8 answers for each question (temperature=0.6) and calculates the average accuracy. Detailed evaluation code can be found in [R1-Evaluation](https://github.com/zpqiu/R1-Evaluation).


|  | AIME 2024 | MATH 500 | GPQA-Diamond |
| --- | --- | --- | --- |
| Qwen2.5-VL-7B-Instruct | 5.21 | 64.55 | 30.68 |
| Qwen2.5-VL-7B-Instruct-RL-Multimodal | 4.79 | 65.30 | 45.45 |

### Multimodal Task Evaluation

Multimodal evaluation covers various types of datasets:

- **Mathematics**: MMathVerse, MathVista
- **Chart Understanding**: ChartQA
- **Comprehensive Abilities**: MME (Cognition and Perception)
- **Multi-dimensional Abilities**: MMStar (Science and Technology, Mathematics, Logical Reasoning, Instance Reasoning, Fine-grained Perception, Coarse-grained Perception)

The evaluation uses the **lmm-eval** framework, which is a comprehensive multimodal evaluation tool. An example evaluation command is as follows, for more usage please refer to [lmm-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval):

```bash
echo OPENAI_API_KEY: ${OPENAI_API_KEY}
echo HF_TOKEN: ${HF_TOKEN}
echo HF_HUB_ENABLE_HF_TRANSFER: ${HF_HUB_ENABLE_HF_TRANSFER}
python3 -m lmms_eval \
    --model vllm \
    --model_args model_version=${model_path},enforce_eager=True,max_model_len=32768,threads=32 \
    --tasks ${task} \
    --batch_size 64 \
    --log_samples \
    --log_samples_suffix vllm \
    --verbosity=DEBUG
```
<table>
  <tr>
    <th> </th>
    <th style="text-align: center;">MMathVerse</th>
    <th colspan="3" style="text-align: center;">MathVista</th>
    <th style="text-align: center;">ChartQA</th>
    <th colspan="2" style="text-align: center;">MME</th>
    <th colspan="6" style="text-align: center;">MMStar</th>
  </tr>
  <tr>
    <td> </td>
    <td style="text-align: center;">testmini</td>
    <td style="text-align: center;">testmini_cot</td>
    <td style="text-align: center;">testmini_format</td>
    <td style="text-align: center;">testmini_solution</td>
    <td style="text-align: center;">relax_overall</td>
    <td style="text-align: center;">cognition</td>
    <td style="text-align: center;">perception</td>
    <td style="text-align: center;">science & technology</td>
    <td style="text-align: center;">math</td>
    <td style="text-align: center;">logical reasoning</td>
    <td style="text-align: center;">instance reasoning</td>
    <td style="text-align: center;">fine-grained perception</td>
    <td style="text-align: center;">coarse perception</td>
  </tr>
  <tr>
    <td> Qwen2.5-VL-7B-Instruct </td>
    <td style="text-align: center;">45.2</td>
    <td style="text-align: center;">64.4</td>
    <td style="text-align: center;">56.0</td>
    <td style="text-align: center;">63.7</td>
    <td style="text-align: center;">70.1</td>
    <td style="text-align: center;">1484.1</td>
    <td style="text-align: center;">290.4</td>
    <td style="text-align: center;">22.7</td>
    <td style="text-align: center;">43.9</td>
    <td style="text-align: center;">39.1</td>
    <td style="text-align: center;">54.2</td>
    <td style="text-align: center;">39.2</td>
    <td style="text-align: center;">67.1</td>
  </tr>
  <tr>
    <td> Qwen2.5-VL-7B-Instruct-RL-Multimodal </td>
    <td style="text-align: center;">48.5</td>
    <td style="text-align: center;">70.1</td>
    <td style="text-align: center;">67.4</td>
    <td style="text-align: center;">70.3</td>
    <td style="text-align: center;">86.9</td>
    <td style="text-align: center;">1674.9</td>
    <td style="text-align: center;">634.6</td>
    <td style="text-align: center;">44.6</td>
    <td style="text-align: center;">61.8</td>
    <td style="text-align: center;">62.5</td>
    <td style="text-align: center;">68.3</td>
    <td style="text-align: center;">60.0</td>
    <td style="text-align: center;">72.5</td>
  </tr>
</table>


### Conclusion
- Models trained with multimodal reinforcement learning show improvements over baseline models in both text and multimodal tasks, with greater improvements in multimodal tasks.
- When performing reinforcement learning on multimodal data, the model's response length increases to a limited extent, typically maintaining around 400-500 tokens.
- Using mathematical and scientific multimodal data for reinforcement learning training can effectively improve the model's performance on general multimodal tasks.