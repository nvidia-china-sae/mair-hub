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
Tool Embedding Vector Computation Module
Used to compute embedding vectors for tool descriptions and store them in tool metadata
"""

import os
import json
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from core.base_module import BaseModule
from core.exceptions import ToolDesignError
from utils.file_manager import FileManager
from utils.data_processor import DataProcessor


class ToolEmbedding(BaseModule):
    """Tool Embedding Vector Computer"""
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize tool embedding vector computer
        
        Args:
            config: Configuration dictionary
            logger: Logger
        """
        super().__init__(config, logger)
        
        self.openai_client = None
        self.file_manager = None
        self.data_processor = None
        self.similarity_threshold = 0.9
        self.batch_size = 10
        self.embedding_model = "text-embedding-v4"
        self.embedding_dimensions = 256
        self.api_key = None
        self.base_url = None
        
    def _setup(self):
        """Setup components"""
        from config.settings import settings
        
        # Get embedding configuration from settings
        embedding_config = settings.get_embedding_config()
        
        # Update configuration from settings
        self.api_key = embedding_config.get('api_key')
        self.base_url = embedding_config.get('base_url')
        self.embedding_model = embedding_config.get('model')
        self.embedding_dimensions = embedding_config.get('dimensions')
        self.batch_size = embedding_config.get('batch_size')
        self.similarity_threshold = embedding_config.get('similarity_threshold')
        
        # Initialize OpenAI client
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY not found in environment variables")
            
        self.openai_client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Initialize file manager
        data_path = settings.get_data_path('tools')
        self.file_manager = FileManager(data_path, self.logger)
        
        # Initialize data processor
        self.data_processor = DataProcessor(self.logger)
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        """
        Add embedding vectors to tool data
        
        Args:
            input_data: Dictionary containing tool data file path
            **kwargs: Other parameters
            
        Returns:
            Updated tool list
        """
        try:
            file_path = input_data.get('tools_file_path')
            if not file_path:
                # Use default tools file
                file_path = self._find_latest_tools_file()
            
            if not file_path:
                raise ToolDesignError("No tools file found")
            
            # Load tool data
            tools = self._load_tools_data(file_path)
            
            # Compute embeddings
            updated_tools = self._add_embeddings_to_tools(tools)
            
            # Save updated tool data
            self._save_tools_with_embeddings(updated_tools)
            
            self.logger.info(f"Successfully added embeddings to {len(updated_tools)} tools")
            return updated_tools
            
        except Exception as e:
            self.logger.error(f"Tool embedding processing failed: {e}")
            raise ToolDesignError(f"Failed to process tool embeddings: {e}")
    
    def _find_latest_tools_file(self) -> Optional[str]:
        """Find latest tools file"""
        try:
            tool_files = self.file_manager.list_files(".", "*tools_refined*.json")
            if not tool_files:
                tool_files = self.file_manager.list_files(".", "*tools_batch*.json")
            
            if tool_files:
                # Sort by time, return latest
                return sorted(tool_files)[-1]
            return None
        except Exception as e:
            self.logger.error(f"Failed to find tools file: {e}")
            return None
    
    def _load_tools_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load tool data"""
        try:
            return self.file_manager.load_json(file_path)
        except Exception as e:
            self.logger.error(f"Failed to load tools data: {e}")
            raise ToolDesignError(f"Failed to load tools data: {e}")
    
    def _add_embeddings_to_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add embedding vectors to tools"""
        try:
            # Extract all description texts that need embedding computation
            descriptions = [tool.get('description', '') for tool in tools]
            
            self.logger.info(f"Computing embeddings for {len(descriptions)} tools")
            
            # Batch compute embeddings
            embeddings = self.get_embeddings(descriptions)
            
            # Add embeddings to tool metadata
            updated_tools = []
            for i, tool in enumerate(tools):
                updated_tool = tool.copy()
                if 'metadata' not in updated_tool:
                    updated_tool['metadata'] = {}
                
                updated_tool['metadata']['embedding'] = embeddings[i] if i < len(embeddings) else None
                updated_tool['metadata']['embedding_model'] = self.embedding_model
                updated_tool['metadata']['embedding_updated_at'] = datetime.now().isoformat()
                
                updated_tools.append(updated_tool)
            
            return updated_tools
            
        except Exception as e:
            self.logger.error(f"Failed to add embeddings to tools: {e}")
            raise ToolDesignError(f"Failed to add embeddings: {e}")
    
    def get_embeddings(self, strings: List[str]) -> List[List[float]]:
        """
        Get embedding vectors for strings
        
        Args:
            strings: List of strings
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(strings), self.batch_size):
            batch = strings[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(strings) + self.batch_size - 1) // self.batch_size
            
            self.logger.info(f"Processing embedding batch {batch_num}/{total_batches}")
            
            try:
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch,
                    dimensions=self.embedding_dimensions,
                    encoding_format="float"
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                self.logger.error(f"Error processing embedding batch {batch_num}: {e}")
                # Fill zero vectors for failed batches
                batch_embeddings = [[0.0] * self.embedding_dimensions for _ in batch]
                all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _save_tools_with_embeddings(self, tools: List[Dict[str, Any]]):
        """Save tool data containing embeddings"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tools_with_embeddings_{timestamp}.json"
            
            self.file_manager.save_json(tools, filename)
            
            # Save summary information
            summary = {
                'total_tools': len(tools),
                'embedding_model': self.embedding_model,
                'embedding_dimensions': self.embedding_dimensions,
                'processed_at': timestamp,
                'has_embedding_count': len([t for t in tools if t.get('metadata', {}).get('embedding')])
            }
            
            summary_filename = f"embeddings_summary_{timestamp}.json"
            self.file_manager.save_json(summary, summary_filename)
            
            self.logger.info(f"Saved tools with embeddings to {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save tools with embeddings: {e}")
            raise ToolDesignError(f"Failed to save tools: {e}")