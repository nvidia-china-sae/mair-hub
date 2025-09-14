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
File management utility
Handles file reading, writing, storage and management
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union
import logging
from datetime import datetime

from core.exceptions import DataStorageError


class FileManager:
    """File management utility class"""
    
    def __init__(self, base_dir: Path = None, logger: logging.Logger = None):
        """
        Initialize file manager
        
        Args:
            base_dir: Base directory path
            logger: Logger instance
        """
        self.base_dir = base_dir or Path.cwd()
        self.logger = logger or logging.getLogger(__name__)
    
    def save_json(self, data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
        """
        Save data as JSON file
        
        Args:
            data: Data to save
            file_path: File path
            indent: JSON indentation
        """
        try:
            file_path = Path(file_path)
            if not file_path.is_absolute():
                file_path = self.base_dir / file_path
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save file
            with file_path.open('w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=indent, default=str)
            
            self.logger.debug(f"Saved JSON file: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save JSON file {file_path}: {e}")
            raise DataStorageError(f"Failed to save JSON file: {e}")
    
    def load_json(self, file_path: Union[str, Path]) -> Any:
        """
        Load JSON file
        
        Args:
            file_path: File path
            
        Returns:
            Loaded data
        """
        try:
            file_path = Path(file_path)
            if not file_path.is_absolute():
                file_path = self.base_dir / file_path
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            with file_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.debug(f"Loaded JSON file: {file_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load JSON file {file_path}: {e}")
            raise DataStorageError(f"Failed to load JSON file: {e}")
    
    def save_pickle(self, data: Any, file_path: Union[str, Path]) -> None:
        """
        Save data as pickle file
        
        Args:
            data: Data to save
            file_path: File path
        """
        try:
            file_path = Path(file_path)
            if not file_path.is_absolute():
                file_path = self.base_dir / file_path
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save file
            with file_path.open('wb') as f:
                pickle.dump(data, f)
            
            self.logger.debug(f"Saved pickle file: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save pickle file {file_path}: {e}")
            raise DataStorageError(f"Failed to save pickle file: {e}")
    
    def load_pickle(self, file_path: Union[str, Path]) -> Any:
        """
        Load pickle file
        
        Args:
            file_path: File path
            
        Returns:
            Loaded data
        """
        try:
            file_path = Path(file_path)
            if not file_path.is_absolute():
                file_path = self.base_dir / file_path
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            with file_path.open('rb') as f:
                data = pickle.load(f)
            
            self.logger.debug(f"Loaded pickle file: {file_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load pickle file {file_path}: {e}")
            raise DataStorageError(f"Failed to load pickle file: {e}")
    
    def save_text(self, text: str, file_path: Union[str, Path]) -> None:
        """
        Save text file
        
        Args:
            text: Text content
            file_path: File path
        """
        try:
            file_path = Path(file_path)
            if not file_path.is_absolute():
                file_path = self.base_dir / file_path
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save file
            with file_path.open('w', encoding='utf-8') as f:
                f.write(text)
            
            self.logger.debug(f"Saved text file: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save text file {file_path}: {e}")
            raise DataStorageError(f"Failed to save text file: {e}")
    
    def load_text(self, file_path: Union[str, Path]) -> str:
        """
        Load text file
        
        Args:
            file_path: File path
            
        Returns:
            Text content
        """
        try:
            file_path = Path(file_path)
            if not file_path.is_absolute():
                file_path = self.base_dir / file_path
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            with file_path.open('r', encoding='utf-8') as f:
                text = f.read()
            
            self.logger.debug(f"Loaded text file: {file_path}")
            return text
            
        except Exception as e:
            self.logger.error(f"Failed to load text file {file_path}: {e}")
            raise DataStorageError(f"Failed to load text file: {e}")
    
    def list_files(self, directory: Union[str, Path], pattern: str = "*") -> List[Path]:
        """
        List files in directory
        
        Args:
            directory: Directory path
            pattern: File pattern
            
        Returns:
            List of file paths
        """
        try:
            directory = Path(directory)
            if not directory.is_absolute():
                directory = self.base_dir / directory
            
            if not directory.exists():
                return []
            
            files = list(directory.glob(pattern))
            files = [f for f in files if f.is_file()]
            
            self.logger.debug(f"Found {len(files)} files in {directory}")
            return files
            
        except Exception as e:
            self.logger.error(f"Failed to list files in {directory}: {e}")
            raise DataStorageError(f"Failed to list files: {e}")
    
    def ensure_directory(self, directory: Union[str, Path]) -> Path:
        """
        Ensure directory exists
        
        Args:
            directory: Directory path
            
        Returns:
            Directory path
        """
        try:
            directory = Path(directory)
            if not directory.is_absolute():
                directory = self.base_dir / directory
            
            directory.mkdir(parents=True, exist_ok=True)
            return directory
            
        except Exception as e:
            self.logger.error(f"Failed to create directory {directory}: {e}")
            raise DataStorageError(f"Failed to create directory: {e}")
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get file information
        
        Args:
            file_path: File path
            
        Returns:
            File information dictionary
        """
        try:
            file_path = Path(file_path)
            if not file_path.is_absolute():
                file_path = self.base_dir / file_path
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            stat = file_path.stat()
            
            return {
                "name": file_path.name,
                "path": str(file_path),
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "is_file": file_path.is_file(),
                "is_directory": file_path.is_dir()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get file info for {file_path}: {e}")
            raise DataStorageError(f"Failed to get file info: {e}")
    
    def delete_file(self, file_path: Union[str, Path]) -> None:
        """
        Delete file
        
        Args:
            file_path: File path
        """
        try:
            file_path = Path(file_path)
            if not file_path.is_absolute():
                file_path = self.base_dir / file_path
            
            if file_path.exists():
                file_path.unlink()
                self.logger.debug(f"Deleted file: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to delete file {file_path}: {e}")
            raise DataStorageError(f"Failed to delete file: {e}")
    
    def copy_file(self, source: Union[str, Path], destination: Union[str, Path]) -> None:
        """
        Copy file
        
        Args:
            source: Source file path
            destination: Destination file path
        """
        try:
            import shutil
            
            source = Path(source)
            destination = Path(destination)
            
            if not source.is_absolute():
                source = self.base_dir / source
            if not destination.is_absolute():
                destination = self.base_dir / destination
            
            # Ensure destination directory exists
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(source, destination)
            self.logger.debug(f"Copied file from {source} to {destination}")
            
        except Exception as e:
            self.logger.error(f"Failed to copy file from {source} to {destination}: {e}")
            raise DataStorageError(f"Failed to copy file: {e}") 