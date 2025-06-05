# VLM 强化学习（RL）教程

本教程将介绍如何利用 veRL 框架，结合数学领域的文本数据和多模态数据，对 VLM 模型进行强化学习训练，从而提升其推理能力。主要内容包括：

- [环境安装](#环境安装)
- [数据准备](#数据准备)
- [自定义 Reward Function 与 Prompt Template](#自定义-reward-function-与-prompt-template)
- [训练](#训练)
  - [文本数据训练](#文本数据训练)
  - [多模态数据训练](#多模态数据训练)
  - [混合训练](#混合训练)
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
# install the recommended version
git clone https://github.com/volcengine/verl && cd verl && git checkout ee8c34749df90b88d00439a09a1f2acb51d71bc3 && pip3 install -e .

# 如果需要使用 wandb 来监控实验，需要 wandb login 来登录
wandb login

# 如果使用的 HuggingFace 数据集需要认证信息，则需要 huggingface-cli login 来登录 hf 账户
huggingface-cli login
```
**完成 veRL 环境配置后，请将本目录下的所有文件拷贝至 verl 目录中，以确保后续实验能够正常进行。**

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
        "index": 1,
        "split": "train",
        "answer": "3",
        "question": "Solve this algebra problem: If 3x + 7 = 16, what is the value of x?"
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
        "split": "train",
        "index": 1,
        "answer": "11",
        "question": "<image>Find x. Round to the nearest tenth, if necessary."

    }
}
```

* data_source：标识该样本来源的数据集名称，便于数据追溯与管理。
* prompt：以列表形式存储的对话消息（chat messages）。在训练过程中，框架会自动调用 tokenizer 的 apply_chat_template 方法对其进行处理和分词。
* images：多模态数据中以列表形式存储的图片数据，数据类型为 PIL.Image。图片数量需与 prompt 中 `<image>` 标签出现的次数严格对应。在纯文本数据中，此字段应完全省略。
* ability：描述该样本所涉及的任务类型或能力领域。
* reward_model：在 R1 阶段的 RL 训练中，推荐采用基于规则（rule-based）的 reward function，并在 answer 字段中填写问题的标准答案。
* extra_info：该字段当前未被框架直接使用，可根据实际需求留空或自定义扩展。

### 数据处理

为了确保数据集符合我们预定义的格式规范，需要对原始数据进行预处理和转换。您可以通过执行以下脚本一键完成所有数据集的预处理工作：
```bash
bash process_all_datasets.sh
```

### 数据集介绍

本教程使用了两类数据集进行训练：


#### 文本数据集
文本训练使用 [Skywork-OR1-RL-Data](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data) 数据集，这是一个高质量的纯文本数据集，包含大量数学任务样本。我们的实验表明，即使仅使用纯文本数据训练，也能有效提升多模态模型在视觉任务上的表现。

#### 多模态数据集

多模态训练集使用[MMK12](https://huggingface.co/datasets/FanqingM/MMK12)， 这是手工收集的开源多模态推理数据集，总计约 15K 个样本。

## 自定义 Reward Function 与 Prompt Template

### Reward Function

最新版 veRL 支持通过传入 Python 文件的方式自定义 Reward function。在本实验中，我们采用 [reward_model.py](./src/reward_model.py) 作为奖励函数。

其核心逻辑如下：
- 首先检验模型输出是否符合 \<think\>...\</think\>\<answer\>...\</answer\> 格式，若不符合则返回 -1.0；
- 从 \<answer\> 标签中提取答案，使用规则匹配方式与 ground truth 比对，若匹配成功则返回 1.0；
- 若规则匹配不成功，则调用奖励模型进行答案验证，如果答案与标准答案匹配则返回 1.0，否则返回 -1.0。

### Prompt Template

多模态数据已将提示词与每个样本的问题拼接在一起， 提示词如下所示：

```python
You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>.
```
聊天模版默认使用模型Tokenizer的Chat Template。


## 训练
### 部署奖励模型

单纯依赖基于数学规则的验证作为奖励函数存在明显局限性，容易出现假阴性问题（即答案正确但奖励函数误判为错误）。为了提升奖励验证的准确性，建议用户根据具体应用场景部署合适的奖励模型。例如，[xVerify](https://github.com/IAAR-Shanghai/xVerify) 是一个专为评估推理模型设计的答案验证工具，能够准确提取最终答案并智能比较不同形式的数学表达式等价性。

用户可以使用 vLLM 部署奖励模型，示例命令如下：
```
# 设置奖励模型的服务地址和模型路径
export REWARD_MODEL_URL=http://<your_server_ip>:8888/v1
export REWARD_MODEL_PATH=<your_reward_model_path>
export REWARD_MODEL_NAME=<your_reward_model_name>
# 启动奖励模型服务（以 vLLM 方式部署）
python3 -m vllm.entrypoints.openai.api_server \
  --model ${REWARD_MODEL_PATH} \
  --enable-chunked-prefill \
  --served-model-name ${REWARD_MODEL_NAME} \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port 8888
```
**注：您使用的任何第三方项目或工件都须遵守适用的许可条款，并不在 MAIR-HUB 许可范围内。**

### 启动 RAY

单节点训练，启动ray的命令如下所示：
```bash
ray start --head --node-ip-address 0.0.0.0 --port=6379 --block 
```

多节点训练，启动ray的命令如下所示：
```bash
# 假设有多个节点
# 在主节点执行（MASTER_NODE_IP 为主节点的 IP 地址）
ray start --head --node-ip-address=${MASTER_NODE_IP} --port=6379 --block 

# 在其他节点执行
ray start --address ${MASTER_NODE_IP}:6379 --block 
```

### 文本数据训练

一个有趣的发现是，即使仅使用纯文本数据进行强化学习训练，VLM 模型在视觉任务上的表现也能获得显著提升。这表明，纯文本数据中蕴含的推理模式和知识结构能够迁移到多模态任务中。

我们的实验基于 Qwen2.5-VL-7B-Instruct 模型，运行环境为单机 8 卡。使用纯文本数据集训练时，在主节点执行以下命令：

```bash
# 使用纯文本数据训练
bash run_qwen2.5-vl-7b-instrcut-text.sh
```

### 多模态数据训练

多模态训练的命令与文本训练类似，具体如下：

```bash
# 使用多模态数据训练
bash run_qwen2.5-vl-7b-instrcut-multimodal.sh
```

### 混合训练

实验结果显示，将文本数据与多模态数据结合进行训练，可以显著提升模型的整体表现。这种混合训练方式不仅充分发挥了大规模文本数据在推理和知识迁移方面的优势，同时也融合了多模态数据带来的视觉理解能力，从而使模型在文本理解和视觉推理两方面均获得提升。

需要注意的是，目前 VeRL 尚不支持直接对文本和多模态数据进行混合训练。为了解决这一限制，我们为文本数据样本额外添加了一张空白图片，使其格式与多模态数据保持一致，从而实现了两类数据的统一训练。

```
bash run_qwen2.5-vl-7b-instrcut-text-multimodal.sh
```

## 实验曲线
下图展示了使用混合数据训练过程中奖励值(reward)的变化趋势。随着训练的进行，奖励值稳步上升，表明模型在数学推理任务上的能力不断增强：

![reward 曲线](./assets/reward.png)
下图展示了使用混合数据训练过程中模型回复长度(response length)的变化趋势。可以观察到，随着训练的深入，模型的回复长度有所增加。
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
| Qwen2.5-VL-7B-Instruct | 5.2 | 64.6 | 30.7 |
| Qwen2.5-VL-7B-Instruct-RL-Text | 6.9 | 68.0 | 35.5 |
| Qwen2.5-VL-7B-Instruct-RL-Multimodal | 5.0 | 66.3 | 33.7 |
| Qwen2.5-VL-7B-Instruct-RL-Text-Multimodal | 5.8 | 67.1 | 31.3 |
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
    <td> Qwen2.5-VL-7B-Instruct </td>
    <td style="text-align: center;">46.7</td>
    <td style="text-align: center;">56.0</td>
    <td style="text-align: center;">1484.1</td>
    <td style="text-align: center;">703.6</td>
    <td style="text-align: center;">62.9</td>
  </tr>
  <tr>
    <td> Qwen2.5-VL-7B-Instruct-RL-Text </td>
    <td style="text-align: center;">51.4</td>
    <td style="text-align: center;">68.0</td>
    <td style="text-align: center;">1573.7</td>
    <td style="text-align: center;">690.0</td>
    <td style="text-align: center;">64.6</td>
  </tr>
  <tr>
    <td> Qwen2.5-VL-7B-Instruct-RL-Multimodal </td>
    <td style="text-align: center;">51.5</td>
    <td style="text-align: center;">70.2</td>
    <td style="text-align: center;">1585.9</td>
    <td style="text-align: center;">454.0</td>
    <td style="text-align: center;">62.5</td>
  </tr>
  <tr>
    <td> Qwen2.5-VL-7B-Instruct-RL-Text-Multimodal </td>
    <td style="text-align: center;">51.8</td>
    <td style="text-align: center;">71.4</td>
    <td style="text-align: center;">1603.5</td>
    <td style="text-align: center;">606.4</td>
    <td style="text-align: center;">64.5</td>
  </tr>
</table>

### 结论
- 仅使用纯文本数据进行强化学习训练，模型在视觉任务上也能获得显著提升。
- 在多模态数据上进行强化学习，模型的回复长度增长较为有限，通常保持在400~500 token左右。
- 混合训练能够达到最佳效果。

