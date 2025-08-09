"""
File handler utilities for RD Sharma Question Extractor.

This module provides file operation utilities for JSON, PDF, and other file types.
"""

import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import shutil
import tempfile
import os

from .logger import get_logger
from .exceptions import FileOperationError
from ..config import config

logger = get_logger(__name__)


class FileHandler:
    """Handles file operations for the RD Sharma Question Extractor."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize file handler.
        
        Args:
            base_dir: Base directory for file operations
        """
        self.base_dir = Path(base_dir or config.output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"File handler initialized with base directory: {self.base_dir}")
    
    def save_json(self, data: Any, filename: str, directory: Optional[str] = None) -> str:
        """
        Save data to JSON file.
        
        Args:
            data: Data to save
            filename: Output filename
            directory: Optional subdirectory
            
        Returns:
            Path to saved file
        """
        try:
            output_dir = self.base_dir / (directory or "")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = output_dir / filename
            if not filename.endswith('.json'):
                file_path = file_path.with_suffix('.json')
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"JSON file saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            raise FileOperationError(
                f"Failed to save JSON file: {str(e)}",
                context={"filename": filename, "directory": directory}
            )
    
    def load_json(self, filename: str, directory: Optional[str] = None) -> Any:
        """
        Load data from JSON file.
        
        Args:
            filename: Input filename
            directory: Optional subdirectory
            
        Returns:
            Loaded data
        """
        try:
            input_dir = self.base_dir / (directory or "")
            file_path = input_dir / filename
            if not filename.endswith('.json'):
                file_path = file_path.with_suffix('.json')
            
            if not file_path.exists():
                raise FileOperationError(f"File not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"JSON file loaded: {file_path}")
            return data
            
        except Exception as e:
            raise FileOperationError(
                f"Failed to load JSON file: {str(e)}",
                context={"filename": filename, "directory": directory}
            )
    
    def save_pickle(self, data: Any, filename: str, directory: Optional[str] = None) -> str:
        """
        Save data to pickle file.
        
        Args:
            data: Data to save
            filename: Output filename
            directory: Optional subdirectory
            
        Returns:
            Path to saved file
        """
        try:
            output_dir = self.base_dir / (directory or "")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = output_dir / filename
            if not filename.endswith('.pkl'):
                file_path = file_path.with_suffix('.pkl')
            
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Pickle file saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            raise FileOperationError(
                f"Failed to save pickle file: {str(e)}",
                context={"filename": filename, "directory": directory}
            )
    
    def load_pickle(self, filename: str, directory: Optional[str] = None) -> Any:
        """
        Load data from pickle file.
        
        Args:
            filename: Input filename
            directory: Optional subdirectory
            
        Returns:
            Loaded data
        """
        try:
            input_dir = self.base_dir / (directory or "")
            file_path = input_dir / filename
            if not filename.endswith('.pkl'):
                file_path = file_path.with_suffix('.pkl')
            
            if not file_path.exists():
                raise FileOperationError(f"File not found: {file_path}")
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"Pickle file loaded: {file_path}")
            return data
            
        except Exception as e:
            raise FileOperationError(
                f"Failed to load pickle file: {str(e)}",
                context={"filename": filename, "directory": directory}
            )
    
    def save_text(self, content: str, filename: str, directory: Optional[str] = None) -> str:
        """
        Save text content to file.
        
        Args:
            content: Text content to save
            filename: Output filename
            directory: Optional subdirectory
            
        Returns:
            Path to saved file
        """
        try:
            output_dir = self.base_dir / (directory or "")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = output_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Text file saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            raise FileOperationError(
                f"Failed to save text file: {str(e)}",
                context={"filename": filename, "directory": directory}
            )
    
    def load_text(self, filename: str, directory: Optional[str] = None) -> str:
        """
        Load text content from file.
        
        Args:
            filename: Input filename
            directory: Optional subdirectory
            
        Returns:
            Text content
        """
        try:
            input_dir = self.base_dir / (directory or "")
            file_path = input_dir / filename
            
            if not file_path.exists():
                raise FileOperationError(f"File not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Text file loaded: {file_path}")
            return content
            
        except Exception as e:
            raise FileOperationError(
                f"Failed to load text file: {str(e)}",
                context={"filename": filename, "directory": directory}
            )
    
    def copy_file(self, source_path: str, dest_filename: str, 
                 dest_directory: Optional[str] = None) -> str:
        """
        Copy a file to the output directory.
        
        Args:
            source_path: Source file path
            dest_filename: Destination filename
            dest_directory: Optional destination subdirectory
            
        Returns:
            Path to copied file
        """
        try:
            source_file = Path(source_path)
            if not source_file.exists():
                raise FileOperationError(f"Source file not found: {source_path}")
            
            dest_dir = self.base_dir / (dest_directory or "")
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            dest_path = dest_dir / dest_filename
            
            shutil.copy2(source_file, dest_path)
            
            logger.info(f"File copied: {source_path} -> {dest_path}")
            return str(dest_path)
            
        except Exception as e:
            raise FileOperationError(
                f"Failed to copy file: {str(e)}",
                context={"source": source_path, "destination": dest_filename}
            )
    
    def create_backup(self, file_path: str, backup_suffix: str = ".backup") -> str:
        """
        Create a backup of a file.
        
        Args:
            file_path: Path to file to backup
            backup_suffix: Suffix for backup file
            
        Returns:
            Path to backup file
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileOperationError(f"File not found: {file_path}")
            
            backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
            
            shutil.copy2(file_path, backup_path)
            
            logger.info(f"Backup created: {file_path} -> {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            raise FileOperationError(
                f"Failed to create backup: {str(e)}",
                context={"file_path": str(file_path)}
            )
    
    def calculate_file_hash(self, file_path: str, algorithm: str = "md5") -> str:
        """
        Calculate hash of a file.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm (md5, sha1, sha256)
            
        Returns:
            File hash
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileOperationError(f"File not found: {file_path}")
            
            hash_func = hashlib.new(algorithm)
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            
            return hash_func.hexdigest()
            
        except Exception as e:
            raise FileOperationError(
                f"Failed to calculate file hash: {str(e)}",
                context={"file_path": str(file_path), "algorithm": algorithm}
            )
    
    def list_files(self, directory: Optional[str] = None, 
                  pattern: Optional[str] = None) -> List[str]:
        """
        List files in a directory.
        
        Args:
            directory: Directory to list (relative to base_dir)
            pattern: Optional glob pattern
            
        Returns:
            List of file paths
        """
        try:
            target_dir = self.base_dir / (directory or "")
            
            if not target_dir.exists():
                return []
            
            if pattern:
                files = list(target_dir.glob(pattern))
            else:
                files = list(target_dir.iterdir())
            
            return [str(f) for f in files if f.is_file()]
            
        except Exception as e:
            raise FileOperationError(
                f"Failed to list files: {str(e)}",
                context={"directory": directory, "pattern": pattern}
            )
    
    def delete_file(self, filename: str, directory: Optional[str] = None) -> bool:
        """
        Delete a file.
        
        Args:
            filename: File to delete
            directory: Optional subdirectory
            
        Returns:
            True if deleted successfully
        """
        try:
            file_path = self.base_dir / (directory or "") / filename
            
            if not file_path.exists():
                logger.warning(f"File not found for deletion: {file_path}")
                return False
            
            file_path.unlink()
            logger.info(f"File deleted: {file_path}")
            return True
            
        except Exception as e:
            raise FileOperationError(
                f"Failed to delete file: {str(e)}",
                context={"filename": filename, "directory": directory}
            )
    
    def create_temp_file(self, content: str, suffix: str = ".tmp") -> str:
        """
        Create a temporary file with content.
        
        Args:
            content: Content to write to file
            suffix: File suffix
            
        Returns:
            Path to temporary file
        """
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, 
                                           delete=False, encoding='utf-8') as f:
                f.write(content)
                temp_path = f.name
            
            logger.info(f"Temporary file created: {temp_path}")
            return temp_path
            
        except Exception as e:
            raise FileOperationError(
                f"Failed to create temporary file: {str(e)}"
            )
    
    def cleanup_temp_files(self, temp_files: List[str]):
        """
        Clean up temporary files.
        
        Args:
            temp_files: List of temporary file paths
        """
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.debug(f"Temporary file cleaned up: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
    
    def ensure_directory(self, directory: str) -> str:
        """
        Ensure a directory exists.
        
        Args:
            directory: Directory path (relative to base_dir)
            
        Returns:
            Path to created directory
        """
        try:
            dir_path = self.base_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
            logger.debug(f"Directory ensured: {dir_path}")
            return str(dir_path)
            
        except Exception as e:
            raise FileOperationError(
                f"Failed to ensure directory: {str(e)}",
                context={"directory": directory}
            )
    
    def get_file_info(self, filename: str, directory: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a file.
        
        Args:
            filename: File name
            directory: Optional subdirectory
            
        Returns:
            File information dictionary
        """
        try:
            file_path = self.base_dir / (directory or "") / filename
            
            if not file_path.exists():
                raise FileOperationError(f"File not found: {file_path}")
            
            stat = file_path.stat()
            
            info = {
                "path": str(file_path),
                "size": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "is_file": file_path.is_file(),
                "is_directory": file_path.is_dir(),
                "extension": file_path.suffix,
                "hash_md5": self.calculate_file_hash(str(file_path), "md5")
            }
            
            return info
            
        except Exception as e:
            raise FileOperationError(
                f"Failed to get file info: {str(e)}",
                context={"filename": filename, "directory": directory}
            ) 