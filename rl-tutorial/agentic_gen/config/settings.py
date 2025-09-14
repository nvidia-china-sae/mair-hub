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
Global configuration settings
Contains configuration parameters for all modules
"""

import os
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv


class Settings:
    """Global configuration class"""
    
    def __init__(self):
        # Project root directory
        self.ROOT_DIR = Path(__file__).parent.parent
        self.DATA_DIR = self.ROOT_DIR / "data"
        
        # Load .env file
        env_file = self.ROOT_DIR / ".env"
        if env_file.exists():
            load_dotenv(env_file)
        # API configuration
        self.LLM_CONFIG = {
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                "model": os.getenv("OPENAI_MODEL", "gpt-4"),
                "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
                "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
            },
        }
        # Embedding configuration
        self.EMBEDDING_CONFIG = {
            "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
            "base_url": os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "model": os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
            "dimensions": int(os.getenv("EMBEDDING_DIMENSIONS", "256")),
            "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "10")),
            "similarity_threshold": float(os.getenv("EMBEDDING_SIMILARITY_THRESHOLD", "0.9")),
        }
        
        # Default LLM provider to use
        self.DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
        
        # Data path configuration
        self.DATA_PATHS = {
            "scenarios": self.DATA_DIR / "generated" / "scenarios",
            "tools": self.DATA_DIR / "generated" / "tools", 
            "agents": self.DATA_DIR / "generated" / "agents",
            "tasks": self.DATA_DIR / "generated" / "tasks",
            "user_personas": self.DATA_DIR / "generated" / "user_personas",
            "trajectories": self.DATA_DIR / "generated" / "trajectories",
            "trajectory_evaluations": self.DATA_DIR / "generated" / "trajectory_evaluations",
            "high_quality_trajectories": self.DATA_DIR / "filtered" / "high_quality_trajectories",
            "training_data": self.DATA_DIR / "filtered" / "training_data",
            "temp": self.DATA_DIR / "temp",
            "cache": self.DATA_DIR / "temp" / "cache"
        }
        
        # Generation configuration (supports environment variable override)
        self.GENERATION_CONFIG = {
            "scenarios": {
                "target_count": int(os.getenv("SCENARIO_TARGET_COUNT", "50")),
                # you can add more domains here
                "domains": [
                    "food_delivery",
                    "robot_control",
                    "social_media",
                    "ecommerce",
                    "travel",
                    ],
                "batch_size": int(os.getenv("SCENARIO_BATCH_SIZE", "10"))
            },
            "tools": {
                "tools_per_scenario": int(os.getenv("TOOLS_PER_SCENARIO", "10")),
                "batch_size": int(os.getenv("TOOL_BATCH_SIZE", "5"))
            },
            "agents": {
                "target_count": int(os.getenv("AGENT_TARGET_COUNT", "1000")),
                "tools_per_agent": {
                    "min": int(os.getenv("AGENT_MIN_TOOLS", "3")), 
                    "max": int(os.getenv("AGENT_MAX_TOOLS", "6"))
                },
                # todo
                # "batch_size": int(os.getenv("AGENT_BATCH_SIZE", "50"))
            },
            "tasks": {
                "tasks_per_difficulty": 1,
                # todo
                # "max_workers": 64
            },
            "user_personas": {
                "target_count": int(os.getenv("USER_PERSONA_TARGET_COUNT", "500")),
                "personality_types": [
                    "friendly", "impatient", "cautious", "casual"
                ],
                "interaction_styles": [
                    "formal", "informal", "life_oriented"
                ],
                "batch_size": int(os.getenv("USER_PERSONA_BATCH_SIZE", "25"))
            },
            "trajectories": {
                "max_count": int(os.getenv("TRAJECTORY_MAX_COUNT", "1000")),
                "max_workers": int(os.getenv("TRAJECTORY_MAX_WORKERS", "64")),
                "max_turns": int(os.getenv("TRAJECTORY_MAX_TURNS", "40"))
            }
        }
        
        # Simulator configuration
        self.SIMULATOR_CONFIG = {
            "success_rate": float(os.getenv("SIMULATOR_SUCCESS_RATE", "0.85")),     # Tool execution success rate
            "partial_failure_rate": float(os.getenv("SIMULATOR_PARTIAL_FAILURE_RATE", "0.10")),  # Partial failure rate
            "complete_failure_rate": float(os.getenv("SIMULATOR_COMPLETE_FAILURE_RATE", "0.05")),  # Complete failure rate
            "state_persistence": True  # Whether to persist state
        }
        
        # Logging configuration
        self.LOGGING_CONFIG = {
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file_path": self.ROOT_DIR / "logs" / "agent_data_gen.log",
            "max_size": "10MB",
            "backup_count": 5
        }
        
        # Concurrency configuration
        self.CONCURRENCY_CONFIG = {
            "max_workers": int(os.getenv("MAX_WORKERS", "4")),
            "batch_processing": True,
        }
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary data directories"""
        for path in self.DATA_PATHS.values():
            path.mkdir(parents=True, exist_ok=True)
        
        # Create log directory
        log_dir = self.ROOT_DIR / "logs"
        log_dir.mkdir(exist_ok=True)
    
    def get_llm_config(self, provider: str = None) -> Dict[str, Any]:
        """Get LLM configuration"""
        provider = provider or self.DEFAULT_LLM_PROVIDER
        return self.LLM_CONFIG.get(provider, self.LLM_CONFIG["openai"])
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration"""
        return self.EMBEDDING_CONFIG
    
    def get_data_path(self, data_type: str) -> Path:
        """Get data storage path"""
        return self.DATA_PATHS.get(data_type, self.DATA_DIR)
    
    def update_config(self, section: str, updates: Dict[str, Any]):
        """Update configuration"""
        if hasattr(self, section):
            config = getattr(self, section)
            if isinstance(config, dict):
                config.update(updates)
            else:
                raise ValueError(f"Config section {section} is not a dictionary")
        else:
            raise ValueError(f"Config section {section} does not exist")


# Create global configuration instance
settings = Settings() 