# Async DAPO Recipe

## Overview

Inspired by MiMo, this Async DAPO recipe combines veRL's AgentLoop async rollout functionality with the DAPO algorithm to enhance end-to-end training efficiency. This recipe provides efficient distributed processing with intelligent resource management, early stopping mechanisms, and dynamic task scheduling for single-turn RLVR.

## Key Features

- **🚀 Asynchronous Processing**: Non-blocking concurrent request handling for maximum throughput
- **⚡ Early Stopping Mechanism**: Intelligent early termination when a target number of prompts are completed and validated (with a reward variance)
- **🔄 Dynamic Load Balancing**: Global load balancer with real-time server allocation
- **🎯 Seamless Reward Computation & Prompt Filtering**: Immediate reward computation and filtering for each prompt upon its response completion, without waiting for other prompts

## Usage

### 0. Install veRL

```bash
git clone https://github.com/volcengine/verl
cd verl
# !IMPORTANT: checkout the commit, otherwise there may be incompatibility issues
git checkout ac826e0558017a9c675818f36cbb3473c14a2a50
pip install -e .

# copy the current directory to verl/recipe/async_dapo
cp -r async_dapo verl/recipe/async_dapo
```

### 1. Prepare Data

```bash
export DATA_HOME=${DATA_HOME:-"${HOME}/data"}
bash recipe/async_dapo/prepare_dapo_data.sh
```

### 2. Train

We use Qwen3-8B-Base and the DAPO dataset as the model and data. The main training hyperparameters are from DAPO.

```bash
export VLLM_USE_V1=1 
export HOME_DIR=${HOME}
# Now, we haved entered the verl root directory
bash recipe/async_dapo/test_qwen3_8b.sh
```


## Architecture Overview

```mermaid
graph TB
    subgraph "Agent Loop Manager"
        ALM[Agent Loop Manager]
        GLB[Global Load Balancer]
        ESC[Early Stopping Coordinator]
    end
    
    subgraph "Worker Layer"
        ALW1[Agent Loop Worker 1]
        ALW2[Agent Loop Worker 2]
        ALW3[Agent Loop Worker N...]
    end
    
    subgraph "Server Layer"
        AS1[Async LLM Server 1]
        AS2[Async LLM Server 2]
        AS3[Async LLM Server N...]
    end
    
    ALM --> ALW1
    ALM --> ALW2
    ALM --> ALW3
    
    ALW1 --> GLB
    ALW2 --> GLB
    ALW3 --> GLB
    
    GLB --> AS1
    GLB --> AS2
    GLB --> AS3
    
    ALW1 --> ESC
    ALW2 --> ESC
    ALW3 --> ESC
```

## Core Mechanisms

### 1. Seamless Reward Computation & Prompt Filtering

We fused the reward computation and prompt filtering process with the rollout process, so that the reward computation and prompt filtering can be done immediately after the response generation is completed.

This eliminates the need to wait for the entire batch rollout to complete, and allows for early stopping and prompt filtering to be done in a more timely manner.


### 2. Early Stopping Coordination

The Early Stopping Coordinator provides intelligent termination control to optimize training efficiency and prevent unnecessary computation.

**Key Components:**
- **Global State Tracking**: Monitors completed and validated prompts across all workers
- **Signal Propagation**: Broadcasts early stopping signals to all active workers

**Implementation Flow:**

```mermaid
sequenceDiagram
    participant W as AsyncLoopWorker
    participant ESC as Early Stopping Coordinator
    participant ALM as AgentLoopManager
    
    ALM->>ESC: Initialize(expected_prompt_num)
    loop For each completed prompt
        W->>ESC: report_completion(sample_index, is_valid)
        ESC->>ESC: Check if target reached
        alt Target Reached
            ESC->>W: should_stop_generation() → True
            W->>W: Cancel pending tasks
        else Continue
            ESC->>W: should_stop_generation() → False
        end
    end
```

### 3. Global Load Balancing

The Global Load Balancer implements sophisticated server allocation strategies to maximize resource utilization and minimize request latency.

**Load Balancing Strategies:**

#### Semaphore-Based Capacity Control
```python
# Capacity management with threading.Semaphore
total_capacity = max_loads_per_server * num_servers
semaphore = threading.Semaphore(total_capacity)

# Server allocation with load tracking
def get_server_index():
    semaphore.acquire()  # Wait for available capacity
    min_load_server = min(servers, key=lambda s: s.current_load)
    min_load_server.current_load += 1
    return min_load_server.index

# Update the server load after a task is completed
def release_server_index(server_index):
    current_loads[server_index] -= 1
    semaphore.release()
```

### 4. Dynamic Task Management

Each AgentLoopWorker first creates rollout tasks for max_concurrent_prompts number of prompts, then creates a rollout task for the next prompt only when one prompt completes generation.

In this way, FIFO control is implemented on the task sending side, so that the N response generation tasks for each prompt are completed at approximately the same time, followed by subsequent reward calculation and filtering. Otherwise, multiple prompts might only complete partial response generation and block subsequent processing.

**Task Lifecycle Management:**

```mermaid
stateDiagram-v2
    [*] --> PendingQueue: Tasks Queued
    PendingQueue --> Running: Create Task (up to max_concurrent)
    Running --> Completed: Task Finished
    Running --> Cancelled: Early Stop Signal
    Completed --> PendingQueue: Create New Task
    Cancelled --> [*]
    Completed --> [*]: No More Pending
    
    note right of Running: Max concurrent tasks  configurable limit
```

**Implementation Details:**

```python
# Dynamic task creation with concurrent limit
max_concurrent_tasks = config.max_concurrent_prompts
pending_prompts = list(prompt_groups.items())
pending_tasks = {}

# Create initial batch
for _ in range(min(max_concurrent_tasks, len(pending_prompts))):
    sample_index, group_data = pending_prompts.pop(0)
    task = asyncio.create_task(process_prompt_group(sample_index, group_data))
    pending_tasks[task] = sample_index

# Main processing loop
while pending_tasks:
    # Wait for any task completion
    done, still_pending = await asyncio.wait(
        pending_tasks.keys(), 
        return_when=asyncio.FIRST_COMPLETED
    )
    
    # Process completed tasks and create new ones
    for task in done:
        process_completed_task(task)
        if pending_prompts and not early_stop_triggered:
            create_new_task()
```

## Data Flow Overview

```mermaid
sequenceDiagram
    participant Client as Client/Trainer
    participant ALM as Agent Loop Manager
    participant ESC as Early Stopping Coordinator
    participant GLB as Global Load Balancer
    participant ALW as Agent Loop Worker
    participant ASM as Async Server Manager
    participant AS as Async LLM Server
    
    Note over Client,AS: Async DAPO Data Flow
    
    Client->>ALM: Input prompts batch
    ALM->>ESC: Initialize(expected_prompt_num)
    ALM->>GLB: Reset load balancer
    ALM->>ALM: Split batch by prompt_index
    
    loop For each worker
        ALM->>ALW: Assign prompt groups
    end
    
    par Worker Processing
        loop Dynamic Task Creation
            ALW->>ALW: Create rollout tasks (max_concurrent_prompts)
            
            loop For each prompt
                ALW->>ESC: Check should_stop_generation()
                alt Continue processing
                    ALW->>ASM: Send generation request
                    ASM->>GLB: Get vllm server index
                    GLB-->>ASM: Return vllm server index with min load
                    ASM->>AS: Generate with cancellation
                    AS-->>ASM: Generated response
                    ASM->>GLB: Release server index
                    ASM-->>ALW: Return response
                    ALW->>ALW: Compute reward & filter
                    ALW->>ESC: Report completion(prompt_index, is_valid)
                    ESC->>ESC: Check if the valid prompts of the expected number have been collected.
                else Early stop triggered
                    ESC-->>ALW: should_stop_generation() = True
                    ALW->>ALW: Cancel pending tasks
                end
            end
        end
    and Early Stopping Coordination
        ESC->>ESC: Track completed prompts
        alt Target reached
            ESC->>ALW: Broadcast stop signal
        end
    end
    
    ALW-->>ALM: Return processed outputs
    ALM->>ALM: Merge all worker outputs
    ALM-->>Client: Final output with metrics
    
    Note over Client,AS: Early stopping enables efficient resource utilization
```


## Experiments

The `test_qwen3_8b.sh` script is just a simple example to show how to use this async DAPO recipe.

The following experiments are conducted on multiple nodes and with a larger max_response_length (16k).

![The training curve of Qwen3-8B-Base on DAPO dataset](assets/exp-qwen3-8b.png)

- Green Line: using the above async DAPO recipe
- Red Lines: the baseline, using the original AgentLoop async rollout.

From the above figure, we can see that the async DAPO recipe can achieve a similar performance to the baseline, but with a much lower rollout time.

Overall throughput has improved by about 15%, and the time spent in the rollout phase has decreased by 20%.

## Future Work

- [ ] Add more experiments with different models.
- [ ] Apply the async DAPO recipe to multi-turn Agentic RL tasks.


## References and Acknowledgments

- [MiMo](https://arxiv.org/abs/2505.07608): We implement the early stopping mechanism and seamless reward computation & filtering based on MiMo's design.
- [veRL](https://github.com/volcengine/verl): We use veRL's AgentLoop async rollout functionality as the baseline and the training framework.
- [DAPO](https://github.com/BytedTsinghua-SIA/DAPO): We use the DAPO dataset as the data.
- [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base): We use Qwen3-8B-Base as the model.
- [Irvingwangjr/verl](https://github.com/Irvingwangjr/verl): For the generation cancellation mechanism in the AsyncvLLMServer, we refer to the implementation in this repo.