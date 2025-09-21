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
Tool Designer
Design and generate related tools based on scenarios
"""

from typing import Dict, Any, List
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.base_module import BaseModule
from core.models import Tool, ToolParameter
from core.exceptions import ToolDesignError
from config.prompts.tool_prompts import ToolPrompts
from utils.llm_client import LLMClient
from utils.data_processor import DataProcessor
from utils.file_manager import FileManager


class ToolDesigner(BaseModule):
    """Tool Designer"""
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize tool designer
        
        Args:
            config: Configuration dictionary
            logger: Logger
        """
        super().__init__(config, logger)
        
        self.llm_client = None
        self.data_processor = None
        self.file_manager = None
        self.prompts = ToolPrompts()
        
        self.max_workers = 64
    
    def _setup(self):
        """Setup components"""
        from config.settings import settings
        
        # Initialize LLM client
        llm_config = settings.get_llm_config()
        llm_config['provider'] = settings.DEFAULT_LLM_PROVIDER
        self.llm_client = LLMClient(llm_config, self.logger)
        
        # Initialize data processor
        self.data_processor = DataProcessor(self.logger)
        
        # Initialize file manager
        data_path = settings.get_data_path('tools')
        self.file_manager = FileManager(data_path, self.logger)
        
        # Update configuration from config parameter
        config = self.config or {}
        self.max_workers = config.get('max_workers', 64) 
        
        self.logger.info(f"ToolDesigner configured with {self.max_workers} max workers")
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate tools based on scenarios
        
        Args:
            input_data: scenarios
            **kwargs: Other parameters
            
        Returns:
            List of generated tools
        """
        try:
            scenarios = input_data.get('scenarios', [])
            
            if not scenarios:
                raise ToolDesignError("No scenarios provided")
            tools_per_scenario = self.config.get('tools_per_scenario', 5)
            
            # Process all scenarios in parallel
            all_tools = []
            total = len(scenarios)
            finished = 0

            def print_progress(finished, total):
                percent = finished / total * 100
                bar_len = 30
                filled_len = int(bar_len * finished // total)
                bar = '█' * filled_len + '-' * (bar_len - filled_len)
                print(f"\r[Progress] |{bar}| {finished}/{total} scenarios ({percent:.1f}%)", end='', flush=True)

            print_progress(finished, total)

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_scenario = {
                    executor.submit(self._generate_scenario_tools, scenario, tools_per_scenario): scenario 
                    for scenario in scenarios
                }
                
                # Collect results
                for idx, future in enumerate(as_completed(future_to_scenario), 1):
                    scenario = future_to_scenario[future]
                    try:
                        scenario_tools = future.result()
                        all_tools.extend(scenario_tools)
                        self.logger.debug(f"Completed scenario: {scenario.get('name', 'Unknown')}")
                    except Exception as e:
                        self.logger.error(f"Failed to process scenario {scenario.get('name', 'Unknown')}: {e}")
                        continue
                    finished += 1
                    print_progress(finished, total)
            
            # Save generated tools
            self._save_tools(all_tools)
            
            self.logger.info(f"Successfully generated {len(all_tools)} tools from {len(scenarios)} scenarios")
            return all_tools
            
        except Exception as e:
            self.logger.error(f"Tool generation failed: {e}")
            raise ToolDesignError(f"Failed to generate tools: {e}")
    
    def _generate_scenario_tools(self, scenario: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
        """
        Generate tools for specific scenario
        
        Args:
            scenario: Scenario data
            count: Generation count
            
        Returns:
            List of tools
        """
        try:
            tools = []
            batch_size = self.config.get('batch_size', 3)
            # Generate tools based on scenario use cases
            while len(tools) < count:
                batch_count = min(batch_size, count - len(tools))
                batch_tools = self._generate_tool_batch(scenario, batch_count)
                tools.extend(batch_tools)
            
            return tools
            
        except Exception as e:
            self.logger.error(f"Failed to generate tools for scenario {scenario.get('name', 'Unknown')}: {e}")
            return []
    
    def _generate_tool_batch(self, scenario: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
        """
        Generate a batch of tools
        
        Args:
            scenario: Scenario data
            count: Generation count
            
        Returns: 
            List of tools
        """
        try:
            prompt = self._build_tool_generation_prompt(scenario, count)
            response = self.llm_client.generate_completion(prompt)
            tools_data = self.llm_client.parse_json_response(response)
            
            # Process and standardize tool data
            tools = []
            for tool_data in tools_data:
                if self._validate_tool_data(tool_data):
                    processed_tool = self._process_tool_data(tool_data, scenario)
                    tools.append(processed_tool)
            
            self.logger.debug(f"Generated {len(tools)} tools for scenario: {scenario.get('name', 'Unknown')}")
            return tools
            
        except Exception as e:
            self.logger.error(f"Failed to generate tool batch: {e}")
            return []
    
    def _build_tool_generation_prompt(self, scenario: Dict[str, Any], count: int) -> str:
        """
        Build tool generation prompt
        
        Args:
            scenario: Scenario data
            count: Generation count
            
        Returns:
            Prompt string
        """
        return self.prompts.TOOL_GENERATION.format(
            scenario_name=scenario.get('name', ''),
            scenario_description=scenario.get('description', ''),
            scenario_domain=scenario.get('domain', ''),
            scenario_context=scenario.get('context', ''),
            count=count
        )
    
    def _validate_tool_data(self, tool_data: Dict[str, Any]) -> bool:
        """
        Validate tool data
        
        Args:
            tool_data: Tool data
            
        Returns:
            Whether valid
        """
        return self.data_processor.validate_tool(tool_data)
    
    def _process_tool_data(self, tool_data: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and standardize tool data
        
        Args:
            tool_data: Raw tool data
            scenario: Related scenario
            
        Returns:
            Processed tool data
        """
        # Generate unique ID
        tool_id = self.data_processor.generate_id('tool', tool_data)
        
        # Clean text content 
        name = tool_data.get('name', '')
        description = tool_data.get('description', '')
        
        # Process parameters
        parameters = []
        for param_data in tool_data.get('parameters', []):
            parameter = {
                'name': param_data.get('name', ''),
                'type': param_data.get('type', 'string'),
                'description': param_data.get('description', ''),
                'required': param_data.get('required', True),
                'default': param_data.get('default'),
                'enum': param_data.get('enum')
            }
            parameters.append(parameter)
        
        processed_tool = {
            'id': tool_id,
            'name': name,
            'description': description,
            'scenario_ids': [scenario.get('id', '')],
            'parameters': parameters,
            'return_type': tool_data.get('return_type', 'object'),
            'examples': tool_data.get('examples', []),
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'scenario_name': scenario.get('name', ''),
                'domain': scenario.get('domain', ''),
            }
        }
        
        return processed_tool
    
    
    def _save_tools(self, tools: List[Dict[str, Any]]):
        """
        Save generated tools
        
        Args:
            tools: List of tools
        """
        try:
            # Save as JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tools_batch_{timestamp}.json"
            
            self.file_manager.save_json(tools, filename)

            self.logger.info(f"Saved {len(tools)} tools to {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save tools: {e}")
    
    def refine_tool(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize single tool
        
        Args:
            tool: Original tool
            
        Returns:
            Optimized tool
        """
        try:
            prompt = self.prompts.TOOL_REFINEMENT.format(tool_data=tool)
            
            response = self.llm_client.generate_completion(prompt)
            refined_data = self.llm_client.parse_json_response(response)
            
            return refined_data
            
        except Exception as e:
            self.logger.error(f"Failed to refine tool: {e}")
            return tool
    
    def evaluate_tool_quality(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate tool quality
        
        Args:
            tool: Tool data
            
        Returns:
            Evaluation result
        """
        try:
            prompt = self.prompts.TOOL_VALIDATION.format(tool_data=tool)
            
            response = self.llm_client.generate_completion(prompt)
            evaluation = self.llm_client.parse_json_response(response)
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate tool quality: {e}")
            return {'overall_score': 3.0, 'suggestions': []}

    def batch_refine_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch optimize tools (multi-threaded version)
        
        Args:
            tools: List of tools
            
        Returns:
            List of optimized tools
        """
        if not tools:
            return []
        
        refined_tools = []
        total = len(tools)
        finished = 0
        
        def print_progress(finished, total):
            percent = finished / total * 100
            bar_len = 30
            filled_len = int(bar_len * finished // total)
            bar = '█' * filled_len + '-' * (bar_len - filled_len)
            print(f"\r[Optimization Progress] |{bar}| {finished}/{total} tools ({percent:.1f}%)", end='', flush=True)
        
        print_progress(finished, total)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_tool = {
                executor.submit(self.refine_tool, tool): tool 
                for tool in tools
            }
            
            # Collect results
            for future in as_completed(future_to_tool):
                tool = future_to_tool[future]
                try:
                    refined_tool = future.result()
                    refined_tools.append(refined_tool)
                except Exception as e:
                    self.logger.error(f"Failed to refine tool {tool.get('name', 'unknown')}: {e}")
                    # If optimization fails, keep original tool
                    refined_tools.append(tool)
                finally:
                    finished += 1
                    print_progress(finished, total)
        
        print()  # New line
        self.logger.info(f"Successfully refined {len(refined_tools)} tools")
        return refined_tools
    
    def batch_evaluate_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch evaluate tool quality
        
        Args:
            tools: List of tools
            
        Returns:
            List of evaluation results
        """
        if not tools:
            return []
        
        evaluations = []
        total = len(tools)
        finished = 0
        
        def print_progress(finished, total):
            percent = finished / total * 100
            bar_len = 30
            filled_len = int(bar_len * finished // total)
            bar = '█' * filled_len + '-' * (bar_len - filled_len)
            print(f"\r[Evaluation Progress] |{bar}| {finished}/{total} tools ({percent:.1f}%)", end='', flush=True)
        
        print_progress(finished, total)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_tool = {
                executor.submit(self.evaluate_tool_quality, tool): tool 
                for tool in tools
            }
            
            # Collect results
            for future in as_completed(future_to_tool):
                tool = future_to_tool[future]
                try:
                    evaluation = future.result()
                    evaluation['id'] = tool.get('id', 'unknown')
                    evaluation['name'] = tool.get('name', 'unknown')
                    evaluations.append(evaluation)
                except Exception as e:
                    self.logger.error(f"Failed to evaluate tool {tool.get('name', 'unknown')}: {e}")
                    # If evaluation fails, add default evaluation
                    evaluations.append({
                        'id': tool.get('id', 'unknown'),
                        'name': tool.get('name', 'unknown'),
                        'overall_score': 3.0,
                        'suggestions': ['Evaluation failed, manual check required'],
                        'error': str(e)
                    })
                finally:
                    finished += 1
                    print_progress(finished, total)
        
        print()  # New line
        self.logger.info(f"Successfully evaluated {len(evaluations)} tools")
        return evaluations
    
    def analyze_evaluation_results(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze batch evaluation results
        
        Args:
            evaluations: List of evaluation results
            
        Returns:
            Analysis and statistics results
        """
        if not evaluations:
            return {}
        
        total_count = len(evaluations)
        scores = [eval_result.get('overall_score', 0) for eval_result in evaluations if 'overall_score' in eval_result]
        
        if not scores:
            return {'total_count': total_count, 'error': 'No valid scores found'}
        
        avg_score = sum(scores) / len(scores)
        
        # Score distribution
        score_distribution = {
            'excellent': len([s for s in scores if s >= 4.5]),
            'good': len([s for s in scores if 4.0 <= s < 4.5]),
            'average': len([s for s in scores if 3.0 <= s < 4.0]),
            'poor': len([s for s in scores if s < 3.0])
        }
        
        # Count recommendation status
        recommendations = {}
        for eval_result in evaluations:
            rec = eval_result.get('recommendation', 'unknown')
            recommendations[rec] = recommendations.get(rec, 0) + 1
        
        return {
            'total_count': total_count,
            'average_score': round(avg_score, 2),
            'min_score': min(scores),
            'max_score': max(scores),
            'score_distribution': score_distribution,
            'recommendations': recommendations,
            'quality_summary': {
                'high_quality_ratio': round((score_distribution['excellent'] + score_distribution['good']) / total_count * 100, 1),
                'needs_improvement_ratio': round(score_distribution['poor'] / total_count * 100, 1)
            }
        }

    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get tool generation statistics
        
        Returns:
            Statistics information
        """
        return self.get_tool_stats()
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """
        Get tool generation statistics
        
        Returns:
            Statistics information
        """
        try:
            tool_files = self.file_manager.list_files(".", "tools_batch_*.json")
            
            total_tools = 0
            domains = set()
            
            for file_path in tool_files:
                tools = self.file_manager.load_json(file_path)
                total_tools += len(tools)
                
                for tool in tools:
                    domains.add(tool.get('metadata', {}).get('domain', ''))
            
            return {
                'total_tools': total_tools,
                'total_domains': len(domains),
                'batch_files': len(tool_files),
                'domains_list': list(domains)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get tool stats: {e}")
            return {} 