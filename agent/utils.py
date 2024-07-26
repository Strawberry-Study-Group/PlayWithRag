import os
import logging
import json
from typing import Any, Dict

def setup_logging(log_file: str, log_level: int = logging.INFO) -> None:
    """
    Set up logging configuration.

    Args:
        log_file (str): Path to the log file.
        log_level (int): Logging level (default: logging.INFO).
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: Loaded data as a dictionary.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to a JSON file.

    Args:
        data (Dict[str, Any]): Data to be saved as a dictionary.
        file_path (str): Path to the JSON file.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def create_directory(directory: str) -> None:
    """
    Create a directory if it doesn't exist.

    Args:
        directory (str): Path to the directory.
    """
    os.makedirs(directory, exist_ok=True)
