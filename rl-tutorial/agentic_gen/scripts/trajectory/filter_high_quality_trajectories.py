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

#!/usr/bin/env python3
"""
High-Quality Trajectory Filter Script

"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from utils.logger import setup_logger
from utils.file_manager import FileManager


def setup_filter_logger():
    """Set up logger for high-quality trajectory filtering"""
    logger = setup_logger(
        "trajectory_filter",
        level=settings.LOGGING_CONFIG["level"],
        log_file=settings.LOGGING_CONFIG["file_path"]
    )
    return logger


def filter_high_quality_trajectories(
    source_dir: Path, 
    target_dir: Path, 
    score_threshold: float,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Filter and copy high-quality trajectory files
    
    Args:
        source_dir: Source directory (trajectory_evaluations)
        target_dir: Target directory (high_quality_trajectories)
        score_threshold: Score threshold
        logger: Logger instance
        
    Returns:
        Filtering result statistics
    """
    logger.info(f"Starting high-quality trajectory filtering: {source_dir} -> {target_dir}")
    logger.info(f"Score threshold: > {score_threshold}")
    
    # Check source directory
    if not source_dir.exists():
        logger.error(f"Source directory does not exist: {source_dir}")
        return {}
    
    # Ensure target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Target directory prepared: {target_dir}")
    
    json_files = list(source_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} evaluation files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                        
            if not isinstance(data, dict):
                logger.warning(f"Skipping invalid format file: {json_file.name}")
                continue
            
            score = data.get('score', 0.0)
            if not isinstance(score, (int, float)):
                logger.warning(f"File {json_file.name} has invalid score format: {score}")
                continue
            
            if float(score) > score_threshold:
                target_file = target_dir / json_file.name
                shutil.copy2(json_file, target_file)
                
                logger.debug(f"Copied high-quality trajectory: {json_file.name} (score: {score:.2f})")
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed {json_file.name}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to process file {json_file.name}: {e}")

    return None


def main():
    """Main function"""
    print("🔍 High-Quality Trajectory Filter")
    print("="*60)
    
    # Setup logging
    logger = setup_filter_logger()
    
    try:
        # 1. Setup directory paths
        source_dir = settings.get_data_path('trajectory_evaluations')
        target_dir = settings.get_data_path('high_quality_trajectories')
        
        print(f"📁 Source directory: {source_dir}")
        print(f"📁 Target directory: {target_dir}")
        
        # 2. Execute filtering
        print(f"🔍 Starting high-quality trajectory filtering (score > 4.0)...")
        
        filter_high_quality_trajectories(
            source_dir=source_dir,
            target_dir=target_dir,
            score_threshold=4.0,
            logger=logger
        )
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ User interrupted execution")
        return 1
    except Exception as e:
        logger.error(f"High-quality trajectory filtering failed: {e}")
        print(f"❌ Filtering failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
