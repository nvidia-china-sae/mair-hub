# VLM 强化学习（RL）教程

本教程将介绍如何利用 veRL 框架，结合数学领域的多模态数据，对 VLM 模型进行强化学习训练，从而提升其推理能力。主要内容包括：

- [环境安装](#环境安装)
- [数据准备](#数据准备)
- [自定义 Reward Function 与 Prompt Template](#自定义-reward-function-与-prompt-template)
- [训练](#训练)
- [实验曲线](#实验曲线)
- [评估实验](#评估实验)

## 环境安装

本节将简要介绍 veRL 的安装方法及关键参数。如果您已熟悉 veRL，可直接跳转至 [数据准备](#数据准备) 部分。

### 环境准备

使用 veRL 官方提供镜像并使用下面的命令启动容器：

```bash
# veRL已经支持了 vllm 0.8+ 的版本，可以使用下面的镜像启动
docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN \
 -v </path/need/to/mount>:</path/in/docker> \
 hiyouga/verl:ngc-th2.6.0-cu120-vllm0.8.2
```

登录容器后，安装 verl 和必要的依赖项

```bash
# install the nightly version (recommended)
git clone https://github.com/volcengine/verl && cd verl && pip3 install -e .

# 我们后续的实验是在数学任务上进行的，因此还需安装 math-verify 来作为奖励函数
pip install math-verify[antlr4_9_3]

# 如果需要使用 wandb 来监控实验，需要 wandb login 来登录
wandb login

# 如果使用的 HuggingFace 数据集需要认证信息，则需要 huggingface-cli login 来登录 hf 账户
huggingface-cli login
```


## 数据准备

### 数据格式

veRL 目前仅支持 parquet 格式的数据文件，并且数据需满足特定的结构要求。

在 veRL 的 RL 阶段，每个样本应遵循如下的数据格式：

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

* data_source：标识该样本来源的数据集名称，便于数据追溯与管理。
* prompt：以列表形式存储的对话消息（chat messages）。在训练过程中，框架会自动调用 tokenizer 的 apply_chat_template 方法对其进行处理和分词。
* images：以列表形式存储的图片数据，数据类型为 bytes。图片数量需与 prompt 中 `<image>` 标签出现的次数严格对应。
* ability：描述该样本所涉及的任务类型或能力领域。
* reward_model：在 R1 阶段的 RL 训练中，推荐采用基于规则（rule-based）的 reward function，并在 answer 字段中填写问题的标准答案。
* extra_info：该字段当前未被框架直接使用，可根据实际需求留空或自定义扩展。

### 数据处理

我们需要根据上述数据格式为数据集编写相应的处理脚本。具体实现可以参考 veRL 官方提供的 [geometry3k 处理脚本](https://github.com/volcengine/verl/blob/main/examples/data_preprocess/geo3k.py)。

### 数据集介绍

本教程所用训练集整合自多个开源数据集，包括 [geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k)、[MathVision](https://huggingface.co/datasets/MathLLMs/MathVision)、[polymath](https://huggingface.co/datasets/him1411/polymath)、[SceMQA-main](https://huggingface.co/datasets/Haozy/SceMQA-main) 以及 [We-Math](https://huggingface.co/datasets/We-Math/We-Math)。总计约 11K 个样本，涵盖图像、文本及图文结合的各类数学问题。

## 自定义 Reward Function 与 Prompt Template

### Reward Function

最新版 veRL 支持通过传入 Python 文件的方式自定义 Reward function。在本实验中，我们采用 [math_verify_for_dapo.py](./math_verify_for_dapo.py) 作为奖励函数。

其核心逻辑如下：
- 从模型输出中提取被 `boxed` 包裹的答案，若未检测到则返回 -1；
- 利用 math-verify 工具对提取出的答案与 ground truth 进行比对，一致则返回 1，否则返回 0。

### Prompt Template

多模态数据已将提示词与每个样本的问题拼接在一起， 提示词如下所示：

```python
Let's think step by step and output the final answer within \\boxed{}.
```
聊天模版默认使用模型Tokenizer的Chat Template。



## 训练

### 启动 RAY

单节点训练， 启动ray的命令如下所示：
```bash
ray start --head --node-ip-address 0.0.0.0 --port=6379 --block 
```

多节点训练， 启动ray的命令如下所示：
```bash
# 假设有4个节点
# 在主节点执行
ray start --head --node-ip-address=${MASTER_NODE_IP} --port=6379 --block 

# 在其他节点执行
ray start --address ${MASTER_NODE_IP}:6379 --block 
```

### 启动训练

我们的实验基于Qwen2.5-VL-7B-Instruct模型进行，运行环境为单机8卡。使用本文介绍的数学多模态数据进行训练后，模型效果相比基线有明显提升。在主节点运行以下命令启动训练：

```bash
# 使用多模态数据训练
bash run_qwen2.5-vl-7b-instrcut-multimodal.sh
```

## 实验曲线
下图展示了训练过程中奖励值(reward)的变化趋势。随着训练的进行，奖励值稳步上升，表明模型在数学推理任务上的能力不断增强：

![reward 曲线](./assets/reward.png)
下图展示了训练过程中模型回复长度(response length)的变化趋势。可以观察到，随着训练的深入，模型的回复长度有所增加，但增长幅度较为适中，通常稳定在400 tokens左右，避免了过度冗长的输出：
![response length 曲线](./assets/response-length.png)




## 评估实验

### 文本任务评估

对于文本任务的评估，我们选取了3个广泛用于推理模型评估的基准数据集：
- **AIME 2024**：美国数学邀请赛题目集，难度较高
- **MATH 500**：包含各类数学问题的综合测试集
- **GPQA-Diamond**：博士级别的物理、化学及生物领域问题集

评估方法采用Pass@1[8]，即对每个问题生成8个回答(temperature=0.6)，计算平均准确率。详细评估代码可参考[R1-Evaluation](https://github.com/zpqiu/R1-Evaluation)。


|  | AIME 2024 | MATH 500 | GPQA-Diamond |
| --- | --- | --- | --- |
| Qwen2.5-VL-7B-Instruct | 5.21 | 64.55 | 30.68 |
| Qwen2.5-VL-7B-Instruct-RL-Multimodal | 4.79 | 65.30 | 45.45 |

### 多模态任务评估

多模态评估覆盖了多种类型的数据集：

- **数学类**：MMathVerse、MathVista
- **图表理解**：ChartQA
- **综合能力**：MME（认知与感知）
- **多维度能力**：MMStar（科学技术、数学、逻辑推理、实例推理、细粒度感知、粗粒度感知）

评估采用**lmm-eval**框架，这是一个功能全面的多模态评估工具。评估命令示例如下，更多用法请参考[lmm-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)：

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


### 结论
- 使用多模态强化学习训练的模型相比基线模型在文本任务和多模态任务上都有提升，但在多模态任务上提升幅度更大。
- 在多模态数据上进行强化学习，模型的回复长度增长较为有限，通常保持在400~500 token左右。
- 使用数学和科学类的多模态数据进行强化学习训练，能够有效提升模型在通用多模态任务上的表现。

