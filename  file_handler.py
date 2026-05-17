"""
File handling utilities for screenshots and data management
"""

import os
import cv2
from datetime import datetime
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class FileHandler:
    """Handles file I/O operations"""

    def __init__(self, screenshot_dir: str = "screenshots", cache_dir: str = ".cache"):
        """
        Initialize file handler

        Args:
            screenshot_dir: Directory for saving screenshots
            cache_dir: Directory for cache files
        """
        self.screenshot_dir = screenshot_dir
        self.cache_dir = cache_dir

        # Create directories if they don't exist
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all required directories exist"""
        for dir_path in [self.screenshot_dir, self.cache_dir]:
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                logger.info(f"Directory ready: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to create directory {dir_path}: {e}")

    def save_screenshot(self, frame: cv2.Mat, prefix: str = "screenshot") -> str:
        """
        Save a frame as PNG screenshot

        Args:
            frame: OpenCV frame to save
            prefix: Filename prefix

        Returns:
            Path to saved file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.png"
            filepath = os.path.join(self.screenshot_dir, filename)

            cv2.imwrite(filepath, frame)
            logger.info(f"Screenshot saved: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save screenshot: {e}")
            return ""

    def load_config(self, filepath: str) -> dict:
        """
        Load JSON configuration file

        Args:
            filepath: Path to config file

        Returns:
            Configuration dictionary
        """
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            logger.info(f"Config loaded: {filepath}")
            return config

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def save_config(self, filepath: str, config: dict):
        """
        Save configuration to JSON file

        Args:
            filepath: Path to save config
            config: Configuration dictionary
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Config saved: {filepath}")

        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def get_screenshot_count(self) -> int:
        """Get total number of screenshots taken"""
        try:
            return len([f for f in os.listdir(self.screenshot_dir) if f.endswith('.png')])
        except Exception:
            return 0

    def clear_cache(self):
        """Clear cache directory"""
        try:
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir)
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")