import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import logging
from .config import config
from .utils import timing

logger = logging.getLogger(__name__)


@timing
def load_dataset(file_path: str) -> pd.DataFrame:
    """Load a single CSV file into a DataFrame"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        logger.info(f"Successfully loaded {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        raise


@timing
def load_multiple_files(file_paths: List[str]) -> Dict[str, pd.DataFrame]:
    """Load multiple CSV files into a dictionary of DataFrames"""
    datasets = {}
    for file_path in file_paths:
        name = Path(file_path).stem
        datasets[name] = load_dataset(file_path)
    return datasets


@timing
def get_train_test_data(train_file: str = None, test_files: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Get training and testing datasets"""
    train_file = train_file or Path(config.RAW_DATA_DIR) / config.TRAIN_FILE
    test_files = test_files or [
        Path(config.RAW_DATA_DIR) / f for f in config.TEST_FILES]

    train_df = load_dataset(train_file)
    test_dfs = load_multiple_files(test_files)

    return train_df, test_dfs


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """Save processed DataFrame to storage"""
    output_path = Path(config.PROCESSED_DATA_DIR) / filename
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")
