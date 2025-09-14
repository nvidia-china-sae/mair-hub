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
Prompt templates for tool execution simulator
"""


class ExecutionPrompts:
    """Tool execution simulator prompt templates"""
    
    TOOL_EXECUTION_SYSTEM = """
You are a **Tool Execution Simulator** within an AI system. Your role is to simulate the execution results of various tools. Based on the tool’s functional description and input parameters, you must generate reasonable and realistic execution outcomes.  

## Core Principles
1. **Authenticity**: The simulated results should accurately reflect how the real tool would behave.  
2. **Consistency**: The same input must always produce the same output (unless the tool’s behavior is inherently random).  
3. **Error Handling**: Appropriately simulate possible errors, warnings, or exceptions.  
4. **State Management**: Maintain continuity of the tool’s execution state across interactions.  

## Execution Result Types
- **Success**: The tool executes normally and returns the expected result.  
- **Partial Success**: The tool executes successfully but with warnings or incomplete information.  
- **Failure**: The execution fails due to issues such as invalid parameters, insufficient permissions, or other errors.  

## Output Format
Always return results strictly in the JSON format defined by the tool’s expected schema.  
```
"""

    EXECUTION_RESULT_TEMPLATE = """
Simulate the execution of the specified tool.  

### Inputs  
- **Tool Call:** {tool_call}  
- **Tool Examples:** {examples}  
- **Current State:** {current_state} 
- **Execution Type:** {execution_type}

### Requirements  
- Verify that parameters are valid and complete.  
- Reflect the tool’s expected behavior and constraints.  
- Appropriately simulate possible errors or exceptions.  
- Update and maintain the system state based on execution.  
- Follow the structure and formatting shown in the provided examples.  
- Please use the execution type to determine the execution result.
- Please refer to the Current State and ensure the generated result is consistent with the current state.

### Output  
Return a realistic execution result strictly in JSON format, consistent with the tool’s schema.  
"""

