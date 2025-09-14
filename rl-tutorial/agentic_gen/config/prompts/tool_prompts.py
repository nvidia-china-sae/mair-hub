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
Prompt templates for tool generation
"""


class ToolPrompts:
    """Tool generation prompt templates"""

    TOOL_GENERATION = """
You are a professional tool designer responsible for creating tools and functions tailored to a given application scenario.

Scenario:
- Name: {scenario_name}
- Description: {scenario_description}
- Domain: {scenario_domain}
- Context: {scenario_context}

Task:
Design exactly {count} tools that are highly relevant to the scenario. Cover both foundational capabilities (e.g., authentication, configuration, data access) and core scenario execution functions. Ensure tools are differentiated, broadly useful, and generalizable.

Strict requirements:
- Use English for all names, descriptions, categories, and examples.
- Tool names and parameter names must be snake_case and unique.
- Parameter types must be one of: string, integer, float, boolean, array, object.
- Return type must be one of: string, integer, float, boolean, array, object.
- Each tool must include exactly:
  1) name (concise, snake_case)
  2) description (1–3 sentences; purpose and when to use)
  3) parameters (array). Each parameter must include:
     - name (snake_case)
     - type (one of allowed types)
     - description (clear and specific; include units or format if relevant)
     - required (boolean)
     - default (null or a type-correct value; if required is true, default must be null)
     - enum (optional; include only if the set of allowed values is small and well-defined)
  4) return_type (one of allowed types)
  5) examples (array with exactly two items):
     - Example 1: a successful call
     - Example 2: an error case (e.g., missing/invalid parameter)
- In both examples:
  - input must be an object that matches the parameters (include all required params for the success example; violate a clear rule for the error example).
  - output must be an object. For success, include at least: result: "success" and data (type-consistent with return_type). For error, include at least: result: "error" and error with code and message; data may be null.
- Do not include placeholders like "TBD" or "lorem ipsum". Avoid secrets. Keep values realistic and consistent with the scenario.
- Ensure no duplicate tools and no contradictory behaviors.
- Validate that examples strictly align with defined parameter types and return types.

Output format:
- Return only a JSON array of tool objects. No prose, no comments, no trailing commas.

JSON structure (template)
[
  {{
    "name": "tool_name",
    "description": "Clear description of what the tool does and when to use it.",
    "parameters": [
      {{
        "name": "param_name",
        "type": "string",
        "description": "What this parameter controls; include format/units if relevant.",
        "required": true,
        "default": null,
        "enum": ["option1", "option2"]
      }}
    ],
    "return_type": "object",
    "examples": [
      {{
        "input": {{"param_name": "example_value"}},
        "output": {{"result": "success", "data": {{"example_field": "value"}}}}
      }},
      {{
        "input": {{"param_name": 123}},
        "output": {{"result": "error", "error": {{"code": "INVALID_TYPE", "message": "param_name must be a string"}}, "data": null}}
      }}
    ]
  }}
]

Final self-check before responding:
- Names and parameters are snake_case and unique.
- Parameter and return types are from the allowed set.
- Required params have default = null; optional params have sensible defaults.
- Examples comply with the parameter schema and return_type.
- Output is valid JSON array with no extra text.
"""

    TOOL_REFINEMENT = """
Please optimize the design of the following tool:

Tool information:
{tool_data}

Optimization requirements:
1. Improve the clarity of tool descriptions
2. Optimize parameter design and type definitions
3. Provide better usage examples
4. Ensure tool practicality
5. Improve parameter validation rules
6. Optimize return data structure

Please return the optimized tool in JSON format, maintaining the original structure:
```json
{{
  "id": "tool_id",
  "name": "tool_name",
  "description": "Optimized tool description",
  "category": "Tool category",
  "scenario_ids": ["scenario_id"],
  "parameters": [
    {{
      "name": "param_name",
      "type": "string",
      "description": "Optimized parameter description",
      "required": true,
      "default": null,
      "enum": ["option1", "option2"],
    }}
  ],
  "return_type": "object",
  "examples": [
    {{
      "input": {{"param_name": "example_value"}},
      "output": {{"result": "success", "data": {{}}}}
    }}
  ],
}}
```
"""

    TOOL_VALIDATION = """
Evaluate the tool’s design quality and usefulness based only on the information below. Do not speculate about missing details.

Tool information:  
{tool_data}

Scoring scale (1=poor, 3=average, 5=excellent):  
- Clarity (clarity): Are descriptions, parameters, and inputs/outputs clear and consistent?  
- Utility (utility): Is the functionality valuable and relevant to typical scenarios/user needs?  
- Usability (usability): Are parameter names/types/defaults/error handling reasonable and easy to invoke?  
- Completeness (completeness): Are required fields, constraints, examples, and edge cases covered?  
- Compliance (compliance): Does it follow design conventions (naming, consistency, type constraints, security/permissions, error codes, etc.)?

Output: return JSON only (no extra text or code block), in this format:  
```json
{{
  "scores": {{ "clarity": n, "utility": n, "usability": n, "completeness": n, "compliance": n }},  
  "overall_score": x.x 
}}
```

Rules:  
- overall_score = average of the five scores, keep one decimal place.  
- If information is missing or ambiguous, deduct in completeness/clarity and note it in weaknesses/suggestions.  
"""
