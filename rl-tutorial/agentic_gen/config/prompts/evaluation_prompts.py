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
Prompt templates for trajectory quality evaluation
"""


class EvaluationPrompts:
    """Trajectory quality evaluation prompt templates"""
    
    TRAJECTORY_EVALUATION_SYSTEM = """
You are an expert evaluator of multi-turn agent interactions. Your task is to assess the quality of conversation trajectories between users and AI agents.

**Evaluation Framework:**

**1. Task Completion Analysis (Weight: 40%)**
- Did the agent fully understand the user's initial request and objectives?
- Were all specified tasks and requirements completed successfully?
- Was the final outcome satisfactory and complete?

**2. Tool Usage Assessment (Weight: 30%)**
- Were the correct tools selected for each step?
- Were tool parameters passed accurately and appropriately?
- Did the agent handle tool execution results correctly?
- Was the sequence of tool calls logical and efficient?
- Did the agent recover gracefully from tool failures?

**3. Interaction Quality Evaluation (Weight: 30%)**
- Was the communication natural and contextually appropriate?
- Did the agent ask for clarification when necessary?
- Were responses timely and relevant?
- Did the agent maintain conversational context throughout?

**Scoring Scale:** 
- 1: Poor (major issues, task failed)
- 2: Below Average (significant problems, partial completion)
- 3: Average (basic completion with some issues)
- 4: Good (completed well with minor issues)
- 5: Excellent (exemplary performance, natural interaction)

Please provide detailed, objective analysis based on the conversation content and tool execution results.
"""

    TRAJECTORY_EVALUATION_USER = """
Please evaluate the following multi-turn agent interaction trajectory:

**Task Information:**
- Task Description: {task_description}
- Success Criteria: {tool_usage_expectations}

**Conversation Trajectory:**
{conversation_history}

**Tool Execution Results:**
{tool_results}

Please provide your evaluation in the following JSON format:

```json
{{
    "overall_score": 0.0,
    "analysis": {{
        "task_completion_analysis": "Detailed analysis of whether the task was completed successfully...",
        "tool_usage_analysis": "Assessment of tool selection, parameter accuracy, and execution handling...",
        "interaction_quality_analysis": "Evaluation of communication quality, context maintenance, and user experience...",
    }},
}}
```

**Evaluation Guidelines:**
1. Be objective and specific in your analysis
2. Focus on measurable outcomes and observable behaviors
3. Consider the user's perspective and satisfaction
"""
