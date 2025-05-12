# VLM 强化学习（RL）教程

本教程将介绍如何利用 veRL 框架，结合数学领域的文本数据和多模态数据，对 VLM 模型进行强化学习训练，从而提升其推理能力。主要内容包括：

- [环境安装](#环境安装)
- [数据准备](#数据准备)
- [自定义 Reward Function 与 Prompt Template](#自定义-reward-function-与-prompt-template)
- [训练](#训练)
  - [文本数据训练](#文本数据训练)
  - [多模态数据训练](#多模态数据训练)
  - [二阶段训练](#二阶段训练)
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

# 如果需要使用 wandb 来监控实验，需要 wandb login 来登录
wandb login

# 如果使用的 HuggingFace 数据集需要认证信息，则需要 huggingface-cli login 来登录 hf 账户
huggingface-cli login
```

### 关键参数说明

关于 veRL 的关键参数说明，请参考 R1-Zero 教程中的[关键参数说明](../r1-zero/README.zh-CN.md#关键参数说明)。

## 数据准备

### 数据格式

veRL 目前仅支持 parquet 格式的数据文件，并且数据需满足特定的结构要求。

#### 纯文本数据格式
对于纯文本数据（如Skywork-OR1），数据格式相似但需省略 images 字段，如下所示：

```json
{
    "data_source": "skywork/OR1",
    "prompt": [{
        "content": "Solve this algebra problem: If 3x + 7 = 16, what is the value of x?   You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>", 
        "role": "user"
    }],
    "ability": "math",
    "reward_model": {
        "style": "rule",
        "ground_truth": "3"
    },
    "extra_info": {
        "answer": "3",
        "split": "train"
    }
}
```


#### 多模态数据格式
在 veRL 的 RL 阶段，每个多模态样本应遵循如下的数据格式：

```json
{
    "data_source": "hiyouga/geometry3k",
    "prompt": [{
        "content": "<image>Find x. Round to the nearest tenth, if necessary. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>", 
        "role": "user"
    }],
    "images": [<PIL.PngImagePlugin.PngImageFile image mode=RGB size=622x406>], 
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
* images：多模态数据中以列表形式存储的图片数据，数据类型为 bytes。图片数量需与 prompt 中 `<image>` 标签出现的次数严格对应。在纯文本数据中，此字段应完全省略。
* ability：描述该样本所涉及的任务类型或能力领域。
* reward_model：在 R1 阶段的 RL 训练中，推荐采用基于规则（rule-based）的 reward function，并在 answer 字段中填写问题的标准答案。
* extra_info：该字段当前未被框架直接使用，可根据实际需求留空或自定义扩展。

### 数据处理

我们需要根据上述数据格式为数据集编写相应的处理脚本。具体实现可以参考 veRL 官方提供的 [geometry3k 处理脚本](https://github.com/volcengine/verl/blob/main/examples/data_preprocess/geo3k.py)。

### 数据集介绍

本教程使用了两类数据集进行训练：



#### 文本数据集
文本训练使用 [Skywork-OR1-RL-Data](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data) 数据集，这是一个高质量的纯文本数据集，包含大量数学任务样本。我们的实验表明，即使仅使用纯文本数据训练，也能有效提升多模态模型在视觉任务上的表现。

#### 多模态数据集
多模态训练集整合自多个开源数据集，包括 [geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k)、[MathVision](https://huggingface.co/datasets/MathLLMs/MathVision)、[polymath](https://huggingface.co/datasets/him1411/polymath)、[SceMQA-main](https://huggingface.co/datasets/Haozy/SceMQA-main) 以及 [We-Math](https://huggingface.co/datasets/We-Math/We-Math)。总计约 11K 个样本，涵盖图像、文本及图文结合的各类数学问题。

## 自定义 Reward Function 与 Prompt Template

### Reward Function

最新版 veRL 支持通过传入 Python 文件的方式自定义 Reward function。在本实验中，我们采用 [xverify_for_dapo.py](./src/xverify_for_dapo.py) 作为奖励函数。

其核心逻辑如下：
- 首先检验模型输出是否符合 <think>...</think><answer>...</answer> 格式，若不符合则返回 -1.0；
- 从 <answer> 标签中提取答案，使用规则匹配方式与 ground truth 比对，若匹配成功则返回 1.0；
- 若规则匹配不成功，则调用 xVerify 模型进行答案验证，如果答案与标准答案匹配则返回 1.0，否则返回 -1.0。

### Prompt Template

多模态数据已将提示词与每个样本的问题拼接在一起， 提示词如下所示：

```python
You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>.
```
聊天模版默认使用模型Tokenizer的Chat Template。

**注：`data`目录下提供的数据已完成处理，可以直接使用。**

## 训练

### 部署 xVerify 模型

[xVerify](https://github.com/IAAR-Shanghai/xVerify) 是一个专为评估推理模型设计的答案验证工具，能够准确提取最终答案并智能比较不同形式的数学表达式等价性。

部署模型：
```bash
python3 -m vllm.entrypoints.openai.api_server --model IAAR-Shanghai/xVerify-0.5B-I --enable-chunked-prefill --served-model-name IAAR-Shanghai/xVerify-0.5B-I --max-model-len 8192 --host 0.0.0.0 --port 8888
```

### 启动 RAY

单节点训练，启动ray的命令如下所示：
```bash
# 设置 xVerify 服务的 URL
export XVERIFY_URL=http://${xVerify_IP}:8888/v1
ray start --head --node-ip-address 0.0.0.0 --port=6379 --block 
```

多节点训练，启动ray的命令如下所示：
```bash
# 假设有多个节点
# 在主节点执行（MASTER_NODE_IP 为主节点的 IP 地址）
export XVERIFY_URL=http://${xVerify_IP}:8888/v1  # xVerify 服务的 URL
ray start --head --node-ip-address=${MASTER_NODE_IP} --port=6379 --block 

# 在其他节点执行
ray start --address ${MASTER_NODE_IP}:6379 --block 
```

### 文本数据训练

一个有趣的发现是，即使仅使用纯文本数据进行强化学习训练，VLM模型在视觉任务上的表现也能获得显著提升。这表明纯文本数据中包含的推理模式和知识结构能够迁移到多模态任务中。

使用 Skywork-OR1 纯文本数据集训练的命令如下：

```bash
# 使用纯文本数据训练
bash run_qwen2.5-vl-3b-instrcut-text.sh
```

### 多模态数据训练

我们的实验基于Qwen2.5-VL-3B-Instruct模型进行，运行环境为单机8卡。使用本文介绍的数学多模态数据进行训练后，模型效果相比基线有明显提升。在主节点运行以下命令启动训练：

```bash
# 使用多模态数据训练
bash run_qwen2.5-vl-3b-instrcut-multimodal.sh
```

### 二阶段训练

我们的实验表明，结合文本和多模态数据进行二阶段训练能够达到最佳效果。具体方法是：
1. 第一阶段：使用 Skywork-OR1 文本数据进行强化学习训练，提升模型的基础推理能力
2. 第二阶段：在第一阶段模型的基础上，使用数学多模态数据进行进一步的强化学习训练(将多模态脚本中MODEL_PATH设置为第一阶段训练的模型路径)

这种方法充分利用了文本数据的规模优势和多模态数据的视觉信息，使模型在文本理解和视觉理解两方面都得到显著提升。

## 实验曲线
下图展示了使用多模态数据训练过程中奖励值(reward)的变化趋势。随着训练的进行，奖励值稳步上升，表明模型在数学推理任务上的能力不断增强：

![reward 曲线](./assets/reward.png)
下图展示了使用多模态数据训练过程中模型回复长度(response length)的变化趋势。可以观察到，随着训练的深入，模型的回复长度有所增加，但增长幅度较为适中，通常稳定在400 tokens左右，避免了过度冗长的输出：
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
| Qwen2.5-VL-3B-Instruct | 2.92 | 59.50 | 27.02 |
| Qwen2.5-VL-3B-Instruct-RL-Text | 5.21 | 68.45 | 25.51 |
| Qwen2.5-VL-3B-Instruct-RL-Multimodal | 3.75 | 62.70 | 40.66 |
| Qwen2.5-VL-3B-Instruct-RL-Text-Multimodal | 3.96 | 63.95 | 34.85 |
### 多模态任务评估

多模态评估覆盖了多种类型的数据集：

- **数学类**：MMathVerse、MathVista
- **综合能力**：MME（认知与感知）
- **多维度能力**：MMStar（科学技术、数学、逻辑推理、实例推理、细粒度感知、粗粒度感知）

评估采用**lmm-eval**框架，这是一个功能全面的多模态评估工具。评估命令示例如下，更多用法请参考[lmm-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)：

```bash
echo OPENAI_API_KEY: ${OPENAI_API_KEY}
echo HF_TOKEN: ${HF_TOKEN}
echo HF_HUB_ENABLE_HF_TRANSFER: ${HF_HUB_ENABLE_HF_TRANSFER}
chat_template="./qwen2vl.jinja" # 在问题后添加了训练中使用的提示词
python3 -m lmms_eval \
    --model vllm \
    --model_args model_version=${model_path},enforce_eager=True,max_model_len=32768,threads=32,chat_template=${chat_template} \
    --tasks ${task} \
    --batch_size 64 \
    --log_samples \
    --log_samples_suffix vllm \
    --verbosity=DEBUG \
```
<table>
  <tr>
    <th> </th>
    <th style="text-align: center;">MMathVerse</th>
    <th style="text-align: center;">MathVista</th>
    <th colspan="2" style="text-align: center;">MME</th>
    <th style="text-align: center;">MMStar</th>
  </tr>
  <tr>
    <td> </td>
    <td style="text-align: center;">testmini</td>
    <td style="text-align: center;">testmini_format</td>
    <td style="text-align: center;">cognition</td>
    <td style="text-align: center;">perception</td>
    <td style="text-align: center;">average</td>
  </tr>
  </tr>
  <tr>
    <td> Qwen2.5-VL-3B-Instruct </td>
    <td style="text-align: center;">29.3</td>
    <td style="text-align: center;">56.6</td>
    <td style="text-align: center;">1138.6</td>
    <td style="text-align: center;">309.6</td>
    <td style="text-align: center;">40.7</td>
  </tr>
  <tr>
    <td> Qwen2.5-VL-3B-Instruct-RL-Text </td>
    <td style="text-align: center;">47.6</td>
    <td style="text-align: center;">62.3</td>
    <td style="text-align: center;">1190.9</td>
    <td style="text-align: center;">272.9</td>
    <td style="text-align: center;">55.6</td>
  </tr>
  <tr>
    <td> Qwen2.5-VL-3B-Instruct-RL-Multimodal </td>
    <td style="text-align: center;">38.5</td>
    <td style="text-align: center;">61.9</td>
    <td style="text-align: center;">1391.8</td>
    <td style="text-align: center;">283.2</td>
    <td style="text-align: center;">56.3</td>
  </tr>
  <tr>
    <td> Qwen2.5-VL-3B-Instruct-RL-Text-Multimodal </td>
    <td style="text-align: center;">44.8</td>
    <td style="text-align: center;">64.1</td>
    <td style="text-align: center;">1469.2</td>
    <td style="text-align: center;">285.3</td>
    <td style="text-align: center;">57.7</td>
  </tr>
</table>

### 结论
- 仅使用纯文本数据进行强化学习训练，模型在视觉任务上也能获得显著提升。
- 使用多模态强化学习训练的模型相比基线模型在文本任务和多模态任务上都有提升，但在多模态任务上提升幅度更大。
- 在多模态数据上进行强化学习，模型的回复长度增长较为有限，通常保持在400~500 token左右。
- 二阶段训练（先文本后多模态）能够达到最佳效果，充分结合了两种数据类型的优势。

