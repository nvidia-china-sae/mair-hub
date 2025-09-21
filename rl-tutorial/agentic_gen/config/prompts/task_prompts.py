# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prompt templates for task generation
"""


class TaskPrompts:
    """Task generation prompt templates"""

    TASK_GENERATION = """
You are an **intelligent task design expert**. Your job is to create a **multi-turn conversation task** for a given AI agent with specific tool capabilities, and to design **evaluation rubrics and checkpoints** for assessing its performance.  

---
**Agent Information:**  
- Available tools: `{available_tools}`  
- Tool details:  
`{tools_details}`  

---
**Task Design Requirements:**  
1. **Multi-Turn Conversation**  
   - The task must require **4–8 conversation turns** to complete.  
   - It should involve **2–4 available tools** in sequential usage (depending on difficulty).  
2. **Capability Match**  
   - Only use tools that the agent currently has access to; do not go beyond its capabilities.  
3. **Difficulty Level** — `{difficulty}` should follow these definitions:  
      - `simple`: 2–3 tools, straightforward flow, minimal steps  
      - `medium`: 3–4 tools, requires conditional reasoning  
      - `complex`: 4–6 tools, involves multi-step coordination and planning  
4. **Realistic Scenario**  
   - The task must be based on real-world or business scenarios with clear practical application.  

---
**Output Format (MUST be in JSON):**
```json
{{
    "task": {{
        "title": "Task title",
        "description": "A detailed second-person-perspective description including the user's role, background, and objectives",
        "difficulty": "{difficulty}",
        "expected_turns": "Expected number of turns (4-8)"
    }},
    "rubric": {{
        "tool_usage_expectations": [
            "Describe expectations for tool usage order and process"
        ],
        "checkpoints": [
            "tool_name(param1=value1, param2=value2)"
        ],
        "success_criteria": [
            "List clear, measurable criteria for task success"
        ]
    }}
}}
```
---
**Important Notes:**  
- The description must be detailed and immersive.  
- Checkpoints should allow objective validation of whether the AI followed correct steps.  
- Success criteria should be specific, measurable, and unambiguous.  

"""