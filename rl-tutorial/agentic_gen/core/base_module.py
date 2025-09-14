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
Base module class definition
Base class for all business modules, providing common functionality
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from .exceptions import AgentDataGenException


class BaseModule(ABC):
    """
    Base class for all business modules
    Provides common functionality and interfaces
    """
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize base module
        
        Args:
            config: Module configuration
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.metadata = {
            "module_name": self.__class__.__name__,
            "created_at": datetime.now(),
            "version": "1.0.0"
        }
        self._initialized = False
        
    def initialize(self) -> None:
        """
        Initialize module
        Subclasses can override this method for specific initialization logic
        """
        try:
            self._setup()
            self._initialized = True
            self.logger.info(f"{self.__class__.__name__} initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.__class__.__name__}: {e}")
            raise AgentDataGenException(f"Module initialization failed: {e}")
    
    def _setup(self) -> None:
        """
        Subclass-specific setup logic
        Subclasses should override this method
        """
        pass
    
    # @abstractmethod
    def process(self, input_data: Any, **kwargs) -> Any:
        """
        Main method for processing input data
        All subclasses must implement this method
        
        Args:
            input_data: Input data
            **kwargs: Additional keyword arguments
            
        Returns:
            Processed data
        """
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data
        Subclasses can override this method for specific validation
        
        Args:
            input_data: Input data to validate
            
        Returns:
            Whether validation passes
        """
        return input_data is not None
    
    def validate_output(self, output_data: Any) -> bool:
        """
        Validate output data
        Subclasses can override this method for specific validation
        
        Args:
            output_data: Output data to validate
            
        Returns:
            Whether validation passes
        """
        return output_data is not None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get module status
        
        Returns:
            Module status information
        """
        return {
            "module_name": self.__class__.__name__,
            "initialized": self._initialized,
            "metadata": self.metadata,
            "config": self.config
        }
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update module configuration
        
        Args:
            new_config: New configuration
        """
        self.config.update(new_config)
        self.logger.info(f"Config updated for {self.__class__.__name__}")
    
    def cleanup(self) -> None:
        """
        Clean up resources
        Subclasses can override this method for specific cleanup logic
        """
        self.logger.info(f"Cleaning up {self.__class__.__name__}")
    
    def __enter__(self):
        """Context manager entry"""
        if not self._initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup() 