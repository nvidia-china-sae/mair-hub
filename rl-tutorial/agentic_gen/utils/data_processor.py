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
Data processing utilities
Provides data conversion, validation, and processing functionalities
"""

import json
import hashlib
from typing import List, Dict, Any, Union, Optional
from datetime import datetime
import logging

from core.models import *
from core.exceptions import ModelValidationError


class DataProcessor:
    """Data processing utility class"""
    
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize data processor
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_scenario(self, scenario_data: Dict[str, Any]) -> bool:
        """
        Validate scenario data
        
        Args:
            scenario_data: Scenario data
            
        Returns:
            Whether validation passes
        """
        try:
            required_fields = ['name', 'description', 'context']
            
            for field in required_fields:
                if field not in scenario_data or not scenario_data[field]:
                    raise ModelValidationError(f"Missing required field: {field}")
            
            # Validate description length
            if len(scenario_data['description']) < 10:
                raise ModelValidationError("Description too short (minimum 10 characters)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Scenario validation failed: {e}")
            return False
    
    def validate_tool(self, tool_data: Dict[str, Any]) -> bool:
        """
        Validate tool data
        
        Args:
            tool_data: Tool data
            
        Returns:
            Whether validation passes
        """
        try:
            required_fields = ['name', 'description']
            
            for field in required_fields:
                if field not in tool_data or not tool_data[field]:
                    raise ModelValidationError(f"Missing required field: {field}")
            
            # Validate parameters
            if 'parameters' in tool_data:
                if not isinstance(tool_data['parameters'], list):
                    raise ModelValidationError("parameters must be a list")
                
                for param in tool_data['parameters']:
                    if not isinstance(param, dict):
                        raise ModelValidationError("Each parameter must be a dict")
                    
                    param_required = ['name', 'type', 'description']
                    for field in param_required:
                        if field not in param:
                            raise ModelValidationError(f"Parameter missing field: {field}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Tool validation failed: {e}")
            return False
    
    def generate_id(self, prefix: str = "", data: Dict[str, Any] = None) -> str:
        """
        Generate unique ID
        
        Args:
            prefix: ID prefix
            data: Data used for hash generation
            
        Returns:
            Unique ID
        """
        if data:
            # Generate hash based on data content
            data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
            hash_value = hashlib.md5(data_str.encode()).hexdigest()[:8]
            return f"{prefix}_{hash_value}" if prefix else hash_value
        else:
            # Generate ID based on timestamp
            timestamp = int(datetime.now().timestamp() * 1000)
            return f"{prefix}_{timestamp}" if prefix else str(timestamp)
    
    def merge_data_batches(self, batches: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Merge data batches
        
        Args:
            batches: List of data batches
            
        Returns:
            Merged data list
        """
        merged_data = []
        for batch in batches:
            if isinstance(batch, list):
                merged_data.extend(batch)
            else:
                merged_data.append(batch)
        
        self.logger.info(f"Merged {len(batches)} batches into {len(merged_data)} items")
        return merged_data
    
    def filter_by_quality(self, data_list: List[Dict[str, Any]], quality_threshold: float = 3.0) -> List[Dict[str, Any]]:
        """
        Filter data by quality
        
        Args:
            data_list: List of data
            quality_threshold: Quality threshold
            
        Returns:
            Filtered data list
        """
        filtered = []
        
        for item in data_list:
            quality_score = item.get('quality_score', 5.0)  # Default full score
            
            if quality_score >= quality_threshold:
                filtered.append(item)
        
        self.logger.info(f"Filtered {len(data_list)} items to {len(filtered)} items (threshold: {quality_threshold})")
        return filtered
    
    def convert_to_model(self, data: Dict[str, Any], model_class) -> Any:
        """
        Convert dictionary data to data model object
        
        Args:
            data: Dictionary data
            model_class: Target model class
            
        Returns:
            Model object
        """
        try:
            # Handle special fields
            if 'created_at' in data and isinstance(data['created_at'], str):
                data['created_at'] = datetime.fromisoformat(data['created_at'])
            
            # Handle enum fields
            if model_class == Task:
                if 'difficulty' in data and isinstance(data['difficulty'], str):
                    data['difficulty'] = DifficultyLevel(data['difficulty'])
                if 'task_type' in data and isinstance(data['task_type'], str):
                    data['task_type'] = TaskType(data['task_type'])
            
            if model_class == UserPersona:
                if 'personality_type' in data and isinstance(data['personality_type'], str):
                    data['personality_type'] = UserPersonalityType(data['personality_type'])
                if 'interaction_style' in data and isinstance(data['interaction_style'], str):
                    data['interaction_style'] = InteractionStyle(data['interaction_style'])
            
            return model_class(**data)
            
        except Exception as e:
            self.logger.error(f"Failed to convert data to {model_class.__name__}: {e}")
            raise ModelValidationError(f"Model conversion failed: {e}")
    
    def convert_model_to_dict(self, model_obj: Any) -> Dict[str, Any]:
        """
        Convert model object to dictionary
        
        Args:
            model_obj: Model object
            
        Returns:
            Dictionary data
        """
        try:
            if hasattr(model_obj, '__dataclass_fields__'):
                data = {}
                for field_name, field_def in model_obj.__dataclass_fields__.items():
                    value = getattr(model_obj, field_name)
                    
                    if isinstance(value, datetime):
                        data[field_name] = value.isoformat()
                    elif isinstance(value, Enum):
                        data[field_name] = value.value
                    elif hasattr(value, '__dataclass_fields__'):
                        data[field_name] = self.convert_model_to_dict(value)
                    elif isinstance(value, list):
                        data[field_name] = [
                            self.convert_model_to_dict(item) if hasattr(item, '__dataclass_fields__') else item 
                            for item in value
                        ]
                    else:
                        data[field_name] = value
                
                return data
            else:
                return model_obj
                
        except Exception as e:
            self.logger.error(f"Failed to convert model to dict: {e}")
            raise ModelValidationError(f"Model conversion failed: {e}")
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: Text 1
            text2: Text 2
            
        Returns:
            Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        # Simple character-level similarity calculation
        from difflib import SequenceMatcher
        
        similarity = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        return similarity
    
    def batch_process(self, data_list: List[Any], processor_func, batch_size: int = 100) -> List[Any]:
        """
        Process data in batches
        
        Args:
            data_list: List of data
            processor_func: Processing function
            batch_size: Batch size
            
        Returns:
            List of processed data
        """
        results = []
        
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            try:
                batch_results = processor_func(batch)
                if isinstance(batch_results, list):
                    results.extend(batch_results)
                else:
                    results.append(batch_results)
            except Exception as e:
                self.logger.error(f"Batch processing failed for batch {i//batch_size + 1}: {e}")
                continue
        
        return results 