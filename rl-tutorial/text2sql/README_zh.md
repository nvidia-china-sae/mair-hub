# Agentic RL - Text2SQL Tutorial

在本 tutorial 中，我们将介绍如何使用 veRL 框架进行 multi-turn 强化学习训练，从而显著提升模型在 Text2SQL 任务上的表现。

## 📋 目录

- [任务介绍](#-任务介绍)
- [数据准备](#️-数据准备)
- [工具定义](#-工具定义)
  - [工具 Schema](#工具-schema)
  - [工具执行类](#工具执行类)
- [奖励函数](#-奖励函数)
- [训练](#-训练)
  - [veRL 参数解释](#verl-参数解释)
  - [启动训练](#启动训练)
  - [训练曲线](#训练曲线)
  - [Show case](#示例案例)
- [实验评估](#-实验评估)
  - [环境配置](#环境配置)
  - [评估脚本](#评估脚本)
- [分析](#-分析)
  - [最后一轮后添加总结](#最后一轮后添加总结)
  - [模型最大轮次对效果的影响](#模型最大轮次对效果的影响)
  - [不同模型对效果的影响](#不同模型对效果的影响)
  - [模型参数量对效果的影响](#模型参数量对效果的影响)

---

## 📖 任务介绍

**Text2SQL** 是一种将自然语言文本（如中文或英文的问题描述）自动转换为可在关系型数据库中执行的 SQL 查询语句的技术。这一任务的目标是让用户能够像聊天一样，用自然语言实现对数据库复杂数据的检索和分析，这大大降低了数据库操作的门槛，使数据分析更加便捷高效。

### 🔄 多轮交互的优势

在实际应用中，Text2SQL 任务不仅可以是单轮（一次输入和一次输出），还可以采用**多轮对话**的形式来完成复杂查询。当模型对用户查询意图或数据库结构不确定时，通过多轮交互可以：

- 🔍 **探索数据库结构**：生成探索性 SQL 查询，获取表结构、字段或部分样例数据
- ❓ **确认用户意图**：就不确定的地方进行二次询问
- ✅ **自动验证SQL**：通过"验证 SQL"自动检查生成语句的正确性和可执行性
- 🔧 **自我纠错**：根据执行结果反馈进行自适应调整

这种能力特别适合处理复杂场景和开放式查询需求，不仅提高了用户体验，也显著增强了 Text2SQL 技术在实际业务中的应用价值。

---

## 🗄️ 数据准备

### 📥 数据集下载

我们使用了 [SkyRL-SQL-653-data](https://huggingface.co/datasets/NovaSky-AI/SkyRL-SQL-653-data) 数据集，这个数据集包含了 653 条高质量的 SQL 数据集。

**步骤 1：下载主数据集**
```bash
huggingface-cli download \
  --repo-type dataset \
  --resume-download \
  NovaSky-AI/SkyRL-SQL-653-data \
  --local-dir SkyRL-SQL-653-data \
  --local-dir-use-symlinks False
```

**步骤 2：下载数据库文件**
```bash
huggingface-cli download seeklhy/OmniSQL-datasets data.zip \
  --repo-type dataset \
  --local-dir <path_to_file.zip>
```

### ⚙️ 数据预处理

下载数据集后，执行预处理脚本：

```bash
python examples/data_preprocess/preprocess_sql_dataset.py \
  --input_file input_file_path \
  --local_dir output_file_path \
  --db_root_path path_to_OmniSQL_data
```

### 📝 Prompt 设计

在 `preprocess_sql_dataset.py` 文件中，我们设计了专门的 prompt，具有以下特点：

1. **多轮对话生成**：要求模型按照多轮对话的方式进行生成，先调用工具探索数据库，待信息完备后，再生成最终回答

2. **结构化输出格式**：
   - 💭 **思考内容**：使用 `<think>` 和 `</think>` 标识
   - 🔧 **工具调用**：使用 `<tool_call>` 和 `</tool_call>` 标识  
   - 📊 **工具结果**：使用 `<tool_response>` 和 `</tool_response>` 标识
   - 🎯 **最终答案**：使用 `<answer>` 和 `</answer>` 标识

**系统 Prompt 定义：**
```python
DEFAULT_SYSTEM_CONTENT = (
    "You are a data science expert. Your task is to understand database schemas and generate valid SQL queries "
    "to answer natural language questions using SQLite database engine. You must conduct reasoning inside "
    "<think> and </think> blocks every time you get new information. After reasoning, you need to explore "
    "or verify database information, you can call a SQL execution tool by <tool_call> execute_sql </tool_call> "
    "and it will return the query results between <tool_response> and </tool_response>. "
    "You can execute SQL queries as many times as you want to explore the database structure and data. "
    "If you find no further exploration is needed, you MUST return your final SQL query enclosed within the <answer> </answer> tags."
)
```

---

## 🔧 工具定义

本节介绍如何使用 veRL 框架配置新工具，主要包括定义工具 Schema 和具体工具类实现。

### 工具 Schema

veRL 中可以使用 YAML 文件定义工具，包含工具的输入、输出等字段信息。在 `examples/sglang_multiturn/config/tool_config/sql_tool_config.yaml` 中，我们定义了 SQL 执行工具：

```yaml
tools:
  - class_name: "verl.tools.sql_tool.SqlTool"
    config:
      # 数据库根路径，包含所有数据集的数据库文件
      db_root_path: "/apps/data/OmniSQL-datasets/data/"
      
      # 并发控制配置
      num_workers: 60                    # Ray worker 数量
      rate_limit: 60                     # 每秒最大请求数
      timeout: 30                        # SQL 执行超时时间（秒）
      num_cpus: 32                       # 并行 SQL 执行的 CPU 数量
      type: native
      
      # 结果截断配置
      max_result_chars: 9000             # 结果字符数截断
      max_result_rows: 50                # 结果行数截断
      
      # 全局限流配置
      enable_global_rate_limit: true     # 是否启用全局限流
      
      # 日志配置
      enable_logging: true               # 是否启用执行日志
      log_dir: "/apps/logs/sql_execution" # 日志存储目录
      
    tool_schema:
      type: function
      function:
        name: execute_sql
        description: Executes SQL queries and returns the results.
        parameters:
          type: object
          properties:
            sql_query:
              type: string
              description: "SQL query to be executed"
          required: 
            - sql_query
```

**配置字段说明：**

| 字段 | 说明 |
|------|------|
| `class_name` | 对应的工具类的具体位置，下面会介绍如何实现该工具类 |
| `config` | 工具执行时的配置，包括数据库文件路径、并发控制、日志配置等 |
| `tool_schema` | 定义 `execute_sql` 函数的输入、输出格式 |

### 工具执行类

在 `verl/tools/sql_tool.py` 中，我们实现了具体的工具类，负责执行模型生成的 SQL 并返回结果。

---

## 🎯 奖励函数

本节介绍我们为 Text2SQL 任务定义的奖励函数机制。

### 📊 奖励规则

| 分数 | 条件 | 说明 |
|------|------|------|
| **-1** | 从最后一轮中解析不出 `<answer>` `</answer>` 之间的内容 | 模型未能提供有效的最终答案 |
| **0** | 可以解析出 SQL 但执行报错，或与 ground truth 答案不一致 | SQL 语法错误或结果不正确 |
| **1** | SQL 执行正确且与 ground truth 答案一致 | 完全正确的解答 |

### 📁 实现细节

关于 Text2SQL 奖励函数的具体实现，请参考 `verl/utils/reward_score/text2sql.py`。

---

## 🚀 训练

### veRL 参数解释

本节介绍 veRL 中与多轮训练相关的关键参数：

| 参数名 | 描述 |
|--------|------|
| `actor_rollout_ref.rollout.name` | 指定本次 rollout 的名称，用于区分不同的训练/评估配置 |
| `actor_rollout_ref.rollout.multi_turn.enable` | 是否启用多轮对话模式。设置为 `true` 时，模型将支持多轮交互，适用于复杂查询场景 |
| `actor_rollout_ref.rollout.multi_turn.max_assistant_turns` | 在多轮对话中，assistant 最多可连续回复的轮数。用于控制对话长度，防止死循环 |
| `actor_rollout_ref.rollout.multi_turn.format` | 多轮对话的输入输出格式规范。可配置为不同的模板，以适配不同的上下文组织方式 |

### 启动训练

训练脚本位于 `examples/sglang_multiturn/run_qwen2.5-7b_text2sql.sh` 文件中。您可以参考该脚本进行训练。

### 训练曲线

<img src="./assets/qwen2.5-7b.png" alt="训练曲线示意图" width="600"/>

通过训练 80 个 steps，我们观察到模型训练趋于收敛：

- 📈 **奖励提升**：模型的 reward 从 0.26 提升到 0.85 左右，在训练样本中模型生成的正确率显著提升
- 📏 **响应长度稳定**：模型生成的 response 长度趋于稳定，由于采用的是非推理模型，生成长度适中
- 🔄 **平均轮次**：收敛后，模型平均需要 2.5 轮来回答一个问题

### 示例案例

<details>
<summary>case 1: 对于复杂问题，模型会将问题拆解成几个步骤，并按照步骤分别去解决问题。</summary>

#### **用户问题**
> Find the names, ticker symbols, 1-year returns, and net expenses of the cream of the crop ETFs, those that are not only the top performers but also have the lowest fees, among the ones that are currently active.

---

#### **第1轮 - 助手思考与行动**

**🧠 思考过程：**
```
To achieve the task, we have to follow these steps:
1. Filter out ETFs that are currently active (is_active = 1).
2. Find the ETFs with the highest 1-year returns (return_1y).
3. For the ETFs found in step 2, find those with the lowest net expenses (net_expense).

First, let's identify ETFs that are currently active.
```

**🔧 工具调用：**
```sql
SELECT etf_id FROM etf_overviews WHERE is_active = 1
```

**📊 执行结果：**
```
 0
 0
 1
```

---

#### **第2轮 - 助手分析与下一步查询**

**💭 分析：** 基于第一个SQL查询，我们得到了两个当前活跃的ETF，etf_id分别为0和1。接下来，我们需要在这些ETF中找出具有最高1年回报率和最低净费用的ETF。

**🔧 工具调用：**
```sql
SELECT MAX(return_1y) as max_return_1y, etf_id 
FROM etf_performance 
WHERE etf_id IN (0, 1) 
GROUP BY etf_id
```

**📊 执行结果：**
```
   0  1
0.05  1
0.05  0
```

---

#### **第3轮 - 进一步筛选费用**

**💭 分析：** 具有最高1年回报率（return_1y = 0.05）的ETF是etf_id为0和1的两个ETF。因此，我们需要不仅按最高1年回报率筛选ETF，还要按最低net_expense进行筛选。

让我们检查1年回报率为0.05的ETF中的最低net_expense。

**🔧 工具调用：**
```sql
SELECT MIN(net_expense) as min_net_expense, etf_id 
FROM etf_fees 
WHERE etf_id IN (0, 1) 
GROUP BY etf_id
```

**📊 执行结果：**
```
   0  1
0.46  1
```

---

#### **第4轮 - 生成最终答案**

**💭 分析：** 在1年回报率为0.05的ETF中，etf_id为1的ETF具有最低的net_expense（0.46）。

基于这个分析，表现最佳的ETF是etf_id为1的那个。让我们获取它的名称、股票代码、1年回报率和净费用。

**🧠 思考过程：**
```
We need to join the relevant tables to retrieve the ETF's full information.
```

**🎯 最终答案：**
```sql
SELECT eo.etf_name, eo.ticker, ep.return_1y, f.net_expense
FROM etf_overviews eo
JOIN etf_fees f ON eo.etf_id = f.etf_id
JOIN etf_performance ep ON eo.etf_id = ep.etf_id
WHERE eo.etf_id = 1 
  AND f.is_current = 1 
  AND ep.return_1y = (SELECT MAX(return_1y) FROM etf_performance WHERE etf_id = 1) 
  AND f.net_expense = (SELECT MIN(net_expense) FROM etf_fees WHERE etf_id = 1)
```

</details>

<details>
<summary>case 2: 对于出错的sql，模型会根据反馈进行修正。</summary>

#### **用户问题**
> Which sailors have the best performance (i.e., the lowest rank) in at least one race of each race type?

---

#### **第1轮 - 初次尝试SQL查询**

**🔧 工具调用：**
```sql
SELECT s.name 
FROM sailors AS s 
JOIN boats AS b ON s.sailor_id = b.helm_id 
JOIN race_results AS rr ON b.boat_id = rr.boat_id 
GROUP BY s.name, rr.rank, rr.race_id, rr.race_type_id 
HAVING COUNT(DISTINCT rr.race_type_id) = (SELECT COUNT(DISTINCT race_type_id) FROM races) 
   AND rr.rank = (SELECT MIN(rank) FROM race_results 
                  WHERE race_id IN (SELECT race_id FROM races) 
                    AND boat_id IN (SELECT boat_id FROM boats WHERE helm_id = s.sailor_id))
```

**❌ 执行结果：**
```
Error executing SQL: no such column: rr.race_type_id
```

**💭 错误分析：** race_results表中不存在race_type_id列，需要通过JOIN获取该信息

---

#### **第2轮 - 修正JOIN关系**

**🔧 工具调用：**
```sql
SELECT s.name 
FROM sailors AS s 
JOIN (SELECT b.boat_id, rr.rank 
      FROM boats AS b 
      JOIN race_results AS rr ON b.boat_id = rr.boat_id) AS subq 
ON s.sailor_id = (SELECT helm_id FROM boats WHERE boat_id = subq.boat_id) 
GROUP BY s.name, subq.rank 
HAVING COUNT(DISTINCT (SELECT race_type_id FROM races WHERE race_id = subq.race_id)) = 
       (SELECT COUNT(DISTINCT race_type_id) FROM races) 
   AND subq.rank = (SELECT MIN(rank) FROM race_results 
                    WHERE race_id IN (SELECT race_id FROM races) 
                      AND boat_id IN (SELECT boat_id FROM boats WHERE helm_id = s.sailor_id))
```

**❌ 执行结果：**
```
Error executing SQL: no such column: subq.race_id
```

**💭 错误分析：** 子查询中缺少race_id字段，需要在子查询中包含更多必要字段

---

#### **第3轮 - 完善子查询结构**

**🔧 工具调用：**
```sql
SELECT s.name 
FROM sailors AS s 
JOIN (SELECT b.boat_id, rr.rank, r.venue_id, rt.race_type_id 
      FROM boats AS b 
      JOIN race_results AS rr ON b.boat_id = rr.boat_id 
      JOIN races AS r ON rr.race_id = r.race_id 
      JOIN race_types AS rt ON r.race_type_id = rt.race_type_id) AS subq 
ON s.sailor_id = (SELECT helm_id FROM boats WHERE boat_id = subq.boat_id) 
GROUP BY s.name, subq.rank, subq.race_type_id 
HAVING COUNT(DISTINCT subq.race_type_id) = (SELECT COUNT(DISTINCT race_type_id) FROM races) 
   AND subq.rank = (SELECT MIN(rank) FROM race_results 
                    WHERE race_id IN (SELECT race_id FROM races) 
                      AND boat_id IN (SELECT boat_id FROM boats WHERE helm_id = s.sailor_id))
```

**✅ 执行结果：**
```
           0
Chris Webber
```

**💭 成功分析：** 通过正确JOIN所有相关表（boats、race_results、races、race_types），成功获取了所需数据

---

#### **第4轮 - 生成最终答案**

**🎯 最终答案：**
```sql
SELECT s.name 
FROM sailors AS s 
JOIN (SELECT b.boat_id, rr.rank, r.venue_id, rt.race_type_id 
      FROM boats AS b 
      JOIN race_results AS rr ON b.boat_id = rr.boat_id 
      JOIN races AS r ON rr.race_id = r.race_id 
      JOIN race_types AS rt ON r.race_type_id = rt.race_type_id) AS subq 
ON s.sailor_id = (SELECT helm_id FROM boats WHERE boat_id = subq.boat_id) 
GROUP BY s.name, subq.rank, subq.race_type_id 
HAVING COUNT(DISTINCT subq.race_type_id) = (SELECT COUNT(DISTINCT race_type_id) FROM races) 
   AND subq.rank = (SELECT MIN(rank) FROM race_results 
                    WHERE race_id IN (SELECT race_id FROM races) 
                      AND boat_id IN (SELECT boat_id FROM boats WHERE helm_id = s.sailor_id))
```

</details>


---

## 📊 实验评估

本节介绍如何进行模型训练后的实验评估。

### 环境配置

为了进行更加准确的离线评估，我们开发了一个完整的评估环境。您可以参考 `sql_eval` 文件夹下的代码，该评估环境具有以下特点：

- ✅ **环境一致性**：评估环境与线上训练时环境完全一致，避免在线和离线的差异
- 📋 **轨迹级验证**：支持轨迹级别的验证分析
- 🔄 **多轨迹采样**：支持同一问题采样 n 条轨迹进行对比

### 评估脚本

**步骤 1: 下载spider 测试集**

数据集下载地址：https://yale-lily.github.io/spider

**步骤 2：启动推理服务器**

首先，需要根据训练后的模型启动一个推理服务器：

```bash
python3 -m sglang.launch_server --model-path <model_path> \
  --host 0.0.0.0 \
  --port 30000 \
  --tp 4 \
  --tool-call-parser qwen25
```

**步骤 3：执行评估**

然后，通过以下命令启动评估：

```bash
python -m main_eval \
  --dataset_path spider_data/filter_test.json \
  --db_root_path spider_data \
  --sample_size 500 \            # 采样多少条数据进行评估
  --model_name <model_name> \
  --n 4                          # 采样n条轨迹
```

---

## 📈 分析

### 分析1: 最后一轮后添加总结

在模型训练时，会设置最大请求轮次。当请求轮次到达上限后，且没有生成最终回复，该样本会被当作负例。

考虑到线上应用时，对于没有生成最终答案的对话，一般会再次请求模型，尽可能生成最终答案。为了保证线上应用和训练时的一致性，我们在训练时添加了一个**最终总结**机制，对于没有生成结果的对话进行总结，并尝试生成一个最终答案。


训练脚本参考：`examples/sglang_multiturn/run_qwen2.5-7b_text2sql_final_summary.sh`

该脚本通过设置 `final_summary` 为 `true` 来启用此功能。


| 模型 | Spider 测试集准确率 |
|----------|-------------------|
| qwen-2.5-7b-instruct | 0.618 |
| without summary | 0.646 |
| with summary | **0.674** |

**🔍 结论：** 通过在训练时添加总结机制，模型在测试集上的表现得到了显著提升。

### 分析2: 模型最大轮次对效果的影响

为了验证最大生成轮次对模型效果的影响，我们进行了对比实验，将 `max_assistant_turn` 设置为 10 轮。

<img src="./assets/qwen2.5-turn10.png" alt="不同最大轮次的训练曲线对比" width="600"/>

> 图中灰色线为最大轮数设置为 10 轮，绿色线为设置为 6 轮的训练曲线。

可以观察到：
- 并不是轮数越多越好
- 对于特定的任务，需要找到最合适的最大生成轮次
- 过多的轮次可能导致训练效率降低而效果未必更好

### 分析2：不同模型对效果的影响

我们对比了 **Qwen-2.5-7B-Instruct** 和 **Qwen-2.5-7B-Instruct-Coder** 两个模型的效果。

<img src="./assets/qwen2.5-coder.png" alt="Qwen-2.5-7B-Instruct-Coder训练曲线" width="600"/>

可以观察到：
- Qwen-2.5-Instruct-Coder 倾向于单轮直接给出最终答案
- Qwen-2.5-7B-Instruct 模型在多轮任务中的表现更好

### 分析4: 模型参数量对效果的影响


我们训练了 **Qwen-2.5-14B-Instruct** 模型，并在测试集上进行评估，与 7B 模型进行对比。

| 模型 | Spider 测试集准确率 | 提升幅度 |
|------|-------------------|----------|
| Qwen-2.5-7B-Instruct | 0.618 | - |
| Qwen-2.5-7B-Ours | 0.646 | +2.8% |
| Qwen-2.5-14B-Instruct | 0.678 | - |
| Qwen-2.5-14B-Ours | **0.788** | **+11.0%** |

可以观察到：
- ✅ 经过训练后，所有模型在测试集上的效果都有提升
- 📈 **14B 模型的效果提升更加明显**（11.0% vs 2.8%）
- 🎯 更大的模型参数量为多轮推理提供了更强的基础能力

---

## 🎉 总结

通过本教程，我们展示了如何使用 veRL 框架进行 Text2SQL 的多轮强化学习训练。这项工作可以为 Text2SQL 任务的实际应用提供有价值的技术路径和实践经验。



## Acknowledgement

本项目基于 veRL 框架的完善能力，以及 SkyRL 团队在Text2sql数据集和技术方案上的贡献。正是有了这些开源工具和社区的共同努力，Text2SQL 多轮强化学习的研究与实践才能顺利推进。在此特别感谢！
