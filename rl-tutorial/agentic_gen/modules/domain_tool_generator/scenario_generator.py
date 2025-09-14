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
Scenario Generator
Responsible for generating diverse application scenarios based on domains
"""

import uuid
from typing import Dict, Any, List
from datetime import datetime
import logging

from core.base_module import BaseModule
from core.models import Scenario
from core.exceptions import ScenarioGenerationError
from config.prompts.scenario_prompts import ScenarioPrompts
from utils.llm_client import LLMClient
from utils.data_processor import DataProcessor
from utils.file_manager import FileManager


class ScenarioGenerator(BaseModule):
    """Scenario Generator"""
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize scenario generator
        
        Args:
            config: Configuration dictionary
            logger: Logger
        """
        super().__init__(config, logger)
        
        self.llm_client = None
        self.data_processor = None
        self.file_manager = None
        self.prompts = ScenarioPrompts()
    
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
        data_path = settings.get_data_path('scenarios')
        self.file_manager = FileManager(data_path, self.logger)
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate scenarios
        
        Args:
            input_data: Dictionary containing domains and target_count
            **kwargs: Other parameters
            
        Returns:
            List of generated scenarios
        """
        try:
            domains = input_data.get('domains', [])
            target_count = input_data.get('target_count', 100)
            
            if not domains:
                raise ScenarioGenerationError("No domains provided")
            
            self.logger.info(f"Generating {target_count} scenarios for {len(domains)} domains")
            
            all_scenarios = []
            scenarios_per_domain = target_count // len(domains)
            
            for domain in domains:
                domain_scenarios = self._generate_domain_scenarios(domain, scenarios_per_domain)
                all_scenarios.extend(domain_scenarios)
            
            # Save generated scenarios
            self._save_scenarios(all_scenarios)
            
            self.logger.info(f"Successfully generated {len(all_scenarios)} scenarios")
            return all_scenarios
            
        except Exception as e:
            self.logger.error(f"Scenario generation failed: {e}")
            raise ScenarioGenerationError(f"Failed to generate scenarios: {e}")
    
    def _generate_domain_scenarios(self, domain: str, count: int) -> List[Dict[str, Any]]:
        """
        Generate scenarios for specific domain
        
        Args:
            domain: Domain name
            count: Generation count
            
        Returns:
            List of scenarios
        """
        scenarios = []
        batch_size = self.config.get('batch_size', 5)
        
        # Loop to generate scenarios until target count is reached
        while len(scenarios) < count:
            batch_count = min(batch_size, count - len(scenarios))
            if batch_count <= 0:
                break
            batch_scenarios = self._generate_scenario_batch(domain, batch_count)
            scenarios.extend(batch_scenarios)
        
        return scenarios[:count]
    
    def _generate_scenario_batch(self, domain: str, count: int) -> List[Dict[str, Any]]:
        """
        Generate a batch of scenarios
        
        Args:
            domain: Domain name
            count: Generation count
            
        Returns:
            List of scenarios
        """
        try:
            prompt = self.prompts.SCENARIO_GENERATION.format(
                domain=domain,
                count=count
            )
            response = self.llm_client.generate_completion(prompt)
            scenarios_data = self.llm_client.parse_json_response(response)
            # Process and standardize scenario data
            scenarios = []
            for scenario_data in scenarios_data:
                if self._validate_scenario_data(scenario_data):
                    processed_scenario = self._process_scenario_data(scenario_data, domain)
                    scenarios.append(processed_scenario)
            self.logger.debug(f"Generated {len(scenarios)} scenarios for {domain}")
            return scenarios
            
        except Exception as e:
            self.logger.error(f"Failed to generate scenario batch for {domain}: {e}")
            return []
    
    def _validate_scenario_data(self, scenario_data: Dict[str, Any]) -> bool:
        """
        Validate scenario data
        
        Args:
            scenario_data: Scenario data
            
        Returns:
            Whether valid
        """
        return self.data_processor.validate_scenario(scenario_data)
    
    def _process_scenario_data(self, scenario_data: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """
        Process and standardize scenario data
        
        Args:
            scenario_data: Raw scenario data
            domain: Domain
            
        Returns:
            Processed scenario data
        """
        # Generate unique ID
        scenario_id = self.data_processor.generate_id('scenario', scenario_data)
        
        # Clean text content
        name = scenario_data.get('name', '')
        description = scenario_data.get('description', '')
        context = scenario_data.get('context', '')
        
        processed_scenario = {
            'id': scenario_id,
            'name': name,
            'description': description,
            'domain': domain,
            'context': context,
            'target_users': scenario_data.get('target_users', []),
            'metadata': {
                'generated_at': datetime.now().isoformat(),
            }
        }
        
        return processed_scenario
    
    def _save_scenarios(self, scenarios: List[Dict[str, Any]]):
        """
        Save generated scenarios
        
        Args:
            scenarios: List of scenarios
        """
        try:
            # Save as JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scenarios_batch_{timestamp}.json"
            
            self.file_manager.save_json(scenarios, filename)
            
            # Save summary information
            summary = {
                'total_count': len(scenarios),
                'domains': list(set(s.get('domain', '') for s in scenarios)),
                'generated_at': timestamp
            }
            
            summary_filename = f"scenarios_summary_{timestamp}.json"
            self.file_manager.save_json(summary, summary_filename)
            
            self.logger.info(f"Saved {len(scenarios)} scenarios to {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save scenarios: {e}")
            
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get generation statistics
        
        Returns:
            Statistics information
        """
        try:
            # Count from saved files
            scenario_files = self.file_manager.list_files(".", "scenarios_batch_*.json")
            
            total_scenarios = 0
            domains = set()
            
            for file_path in scenario_files:
                scenarios = self.file_manager.load_json(file_path)
                total_scenarios += len(scenarios)
                
                for scenario in scenarios:
                    domains.add(scenario.get('domain', ''))
            
            return {
                'total_scenarios': total_scenarios,
                'total_domains': len(domains),
                'batch_files': len(scenario_files),
                'domains_list': list(domains),
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get generation stats: {e}")
            return {} 