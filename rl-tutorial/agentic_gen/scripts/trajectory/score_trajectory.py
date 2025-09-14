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
Trajectory Scoring Script

Load generated trajectory data and perform pre-filtering and quality scoring
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from utils.logger import setup_logger
from utils.file_manager import FileManager
from core.models import Trajectory, InteractionSession, ConversationTurn
from modules.quality_judge import TrajectoryEvaluator
from core.exceptions import QualityEvaluationError


def setup_scoring_logger():
    """Set up logger for trajectory scoring"""
    logger = setup_logger(
        "trajectory_scoring",
        level=settings.LOGGING_CONFIG["level"],
        log_file=settings.LOGGING_CONFIG["file_path"]
    )
    return logger


def validate_environment():
    """Validate environment configuration"""
    required_keys = ['OPENAI_API_KEY']
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        print(f"❌ Missing environment variables: {', '.join(missing_keys)}")
        print("Please ensure the following variables are set in the .env file:")
        for key in missing_keys:
            print(f"  {key}=your_api_key_here")
        return False
    
    print("✅ Environment variable check passed")
    return True


def load_trajectory_files(trajectories_dir: Path, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Load all JSON files from the trajectory directory
    
    Args:
        trajectories_dir: Trajectory data directory
        logger: Logger instance
        
    Returns:
        List of trajectory data
    """
    logger.info(f"Starting to load trajectory files: {trajectories_dir}")
    
    if not trajectories_dir.exists():
        logger.error(f"Trajectory directory does not exist: {trajectories_dir}")
        return []
    
    # Find all JSON files
    json_files = list(trajectories_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files")
    
    trajectories_data = []
    failed_count = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Ensure data contains necessary fields
            if isinstance(data, dict):
                trajectories_data.append(data)
            else:
                logger.warning(f"Skipping invalid format file: {json_file.name}")
                failed_count += 1
                
        except Exception as e:
            logger.error(f"Failed to load file {json_file.name}: {e}")
            failed_count += 1
    
    logger.info(f"Successfully loaded {len(trajectories_data)} trajectory files")
    if failed_count > 0:
        logger.warning(f"Failed to load {failed_count} files")
    
    return trajectories_data


def convert_dict_to_trajectory(trajectory_data: Dict[str, Any]) -> Optional[Trajectory]:
    """Convert trajectory dictionary to Trajectory object"""
    try:
        # Extract basic information
        trajectory_id = trajectory_data.get('trajectory_id') or trajectory_data.get('id')
        if not trajectory_id:
            return None
        
        session_data = {
            'id': trajectory_data.get('session_id', f"{trajectory_id}_session"),
            'task_id': trajectory_data.get('task_id', ''),
            'agent_id': trajectory_data.get('agent_id', ''),
            'turns': trajectory_data.get('messages', trajectory_data.get('turns', [])),
        }
        
        # Convert conversation turns
        turns = []
        turns_data = session_data.get('turns', [])
        
        for turn_data in turns_data:
            if isinstance(turn_data, dict):
                # Handle different data formats
                if 'role' in turn_data:
                    # Training data format
                    speaker_map = {
                        'user': 'user',
                        'assistant': 'agent',
                        'execution': 'execution'
                    }
                    speaker = speaker_map.get(turn_data.get('role'), turn_data.get('role'))
                    message = turn_data.get('content', '')
                    recipient = turn_data.get('recipient', '')
                else:
                    # Original format
                    speaker = turn_data.get('speaker', 'unknown')
                    message = turn_data.get('message', '')
                    recipient = turn_data.get('recipient', '')
                
                turn = ConversationTurn(
                    speaker=speaker,
                    recipient=recipient,
                    message=message,
                    timestamp=turn_data.get('timestamp')
                )
                turns.append(turn)
        
        session = InteractionSession(
            id=session_data.get('id', f"{trajectory_id}_session"),
            task_id=session_data.get('task_id', ''),
            agent_id=session_data.get('agent_id', ''),
            turns=turns,
            metadata=session_data.get('metadata', {})
        )
        
        # Create trajectory object
        trajectory = Trajectory(
            id=trajectory_id,
            session=session,
            created_at=datetime.now()
        )
        
        return trajectory
        
    except Exception as e:
        print(f"⚠️ Failed to convert trajectory {trajectory_data.get('trajectory_id', trajectory_data.get('id', 'unknown'))}: {e}")
        return None


def prefilter_trajectories(
    trajectories: List[Trajectory], 
    evaluator: TrajectoryEvaluator,
    logger: logging.Logger
) -> List[Trajectory]:
    """
    Filter trajectories using pre-filter
    
    Args:
        trajectories: List of trajectories
        evaluator: Trajectory evaluator
        logger: Logger instance
        
    Returns:
        List of trajectories that passed pre-filtering
    """
    logger.info(f"Starting pre-filtering {len(trajectories)} trajectories")
    
    filtered_trajectories = []
    filter_stats = {
        'total': len(trajectories),
        'passed': 0,
        'failed': 0,
        'failure_reasons': {}
    }
    
    for trajectory in trajectories:
        try:
            if evaluator.prefilter_trajectory(trajectory):
                filtered_trajectories.append(trajectory)
                filter_stats['passed'] += 1
            else:
                filter_stats['failed'] += 1
                
        except Exception as e:
            filter_stats['failed'] += 1
    
    pass_rate = filter_stats['passed'] / filter_stats['total'] if filter_stats['total'] > 0 else 0
    logger.info(f"Pre-filtering completed: {filter_stats['passed']}/{filter_stats['total']} passed (pass rate: {pass_rate:.1%})")
    
    return filtered_trajectories


def score_single_trajectory(
    logger: logging.Logger,
    evaluator: TrajectoryEvaluator,
    trajectory: Trajectory
) -> Optional[Dict[str, Any]]:
    """Evaluate single trajectory"""
    try:
        # Execute evaluation
        scored_trajectory = evaluator.evaluate_trajectory(trajectory)
        
        return {
            'trajectory_id': trajectory.id,
            'turns_count': len(trajectory.session.turns),
            'score': scored_trajectory.evaluation_score.overall_score if scored_trajectory.evaluation_score else 0,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Failed to evaluate trajectory - ID: {trajectory.id}, Error: {e}")
        return {
            'trajectory_id': trajectory.id,
            'status': 'failed',
            'error': str(e)
        }

def main():
    """Main function"""
    print("🎯 Trajectory Scorer")
    print("="*60)
    
    # Validate environment
    if not validate_environment():
        return 1
    
    # Setup logging
    logger = setup_scoring_logger()
    
    try:
        # 1. Load trajectory data
        print("📁 Loading trajectory data...")
        
        trajectories_dir = settings.get_data_path('trajectories')
        trajectories_data = load_trajectory_files(trajectories_dir, logger)
        
        if not trajectories_data:
            print("❌ Trajectory data files not found")
            return 1
        
        print(f"✅ Loaded {len(trajectories_data)} trajectory files")
        
        # 2. Convert data format
        print("🔄 Converting trajectory data format...")
        trajectories = []
        
        for traj_data in trajectories_data:
            trajectory = convert_dict_to_trajectory(traj_data)
            if trajectory:
                trajectories.append(trajectory)
        
        valid_trajectories_count = len(trajectories)
        print(f"✅ Successfully converted {valid_trajectories_count} valid trajectories")
        
        if not trajectories:
            print("❌ No valid trajectory data")
            return 1
        
        # 3. Initialize evaluator
        evaluator = TrajectoryEvaluator(logger)
        evaluator.initialize()
        
        print("✅ Evaluator initialization completed")
        
        # 4. Pre-filter trajectories
        print("🔍 Pre-filtering trajectories...")
        filtered_trajectories = prefilter_trajectories(trajectories, evaluator, logger)
        
        if not filtered_trajectories:
            print("❌ No trajectories passed pre-filtering")
            return 1
        print(f"✅ {len(filtered_trajectories)} trajectories passed pre-filtering")

        # 5. Execute scoring
        max_workers = settings.CONCURRENCY_CONFIG.get('max_workers', 4)
        print(f"🎯 Starting trajectory scoring...")
        
        start_time = datetime.now()
        
        scoring_results = []
        successful_count = 0
        failed_count = 0
        
        # Use multi-threading for scoring
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_trajectory = {}
            
            for trajectory in filtered_trajectories:
                future = executor.submit(
                    score_single_trajectory,
                    logger,
                    evaluator,
                    trajectory
                )
                future_to_trajectory[future] = trajectory.id
            
            # Collect scoring results
            for i, future in enumerate(as_completed(future_to_trajectory), 1):
                try:
                    result = future.result()
                    if result:
                        scoring_results.append(result)
                        
                        if result['status'] == 'success':
                            successful_count += 1
                        else:
                            failed_count += 1
                        
                        # Output total progress
                        if i % 10 == 0:
                            print(f"📊 Total progress: {i}/{len(filtered_trajectories)} ({i/len(filtered_trajectories)*100:.1f}%)")
                            
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Trajectory scoring task exception: {e}")
        
        print(f"✅ Scoring completed: {successful_count} trajectories successful, {failed_count} trajectories failed")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ User interrupted execution")
        return 1
    except Exception as e:
        logger.error(f"Trajectory scoring failed: {e}")
        print(f"❌ Scoring failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
