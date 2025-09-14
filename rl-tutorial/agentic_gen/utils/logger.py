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
Logging utility
Unified logging configuration and management
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: str = None,
    max_size: str = "10MB",
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup and configure logger
    
    Args:
        name: Logger name
        level: Log level
        log_file: Log file path
        format_string: Log format string
        max_size: Maximum file size
        backup_count: Number of backup files
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    # Set log level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    logger.setLevel(level_map.get(level.upper(), logging.INFO))
    
    # Set log format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log file is specified)
    if log_file:
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Parse maximum file size
        size_in_bytes = _parse_size(max_size)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=size_in_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def _parse_size(size_str: str) -> int:
    """
    Parse file size string
    
    Args:
        size_str: Size string, e.g. "10MB", "1GB"
        
    Returns:
        Number of bytes
    """
    size_str = size_str.upper().strip()
    
    if size_str.endswith('KB'):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith('MB'):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith('GB'):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    else:
        # Assume it's number of bytes
        return int(size_str)


class ModuleLogger:
    """Module-level logger wrapper class"""
    
    def __init__(self, module_name: str, config: dict = None):
        """
        Initialize module logger
        
        Args:
            module_name: Module name
            config: Logging configuration
        """
        self.module_name = module_name
        self.config = config or {}
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        return setup_logger(
            name=self.module_name,
            level=self.config.get("level", "INFO"),
            log_file=self.config.get("file_path"),
            format_string=self.config.get("format"),
            max_size=self.config.get("max_size", "10MB"),
            backup_count=self.config.get("backup_count", 5)
        )
    
    def debug(self, message: str, **kwargs):
        """Debug level log"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Info level log"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Warning level log"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Error level log"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Critical level log"""
        self.logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Exception log"""
        self.logger.exception(message, **kwargs) 