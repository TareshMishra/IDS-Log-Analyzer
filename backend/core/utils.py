import time
import logging
from functools import wraps
from typing import Any, Callable
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from .config import model_config


def timing(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.getLogger(func.__module__).info(
            f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds"
        )
        return result
    return wrapper


def load_scaler() -> Any:
    """Load the saved scaler"""
    scaler_path = Path(model_config.MODEL_DIR) / model_config.SCALER_NAME
    return joblib.load(scaler_path)


def save_plot(fig, filename: str) -> None:
    """Save matplotlib figure"""
    plots_dir = Path(__file__).parent.parent / "storage/plots"
    plots_dir.mkdir(exist_ok=True)
    fig.savefig(plots_dir / filename, bbox_inches='tight')


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory usage of a DataFrame by downcasting numeric types"""
    start_mem = df.memory_usage().sum() / 1024**2
    logging.info(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    logging.info(f"Memory usage after optimization is {end_mem:.2f} MB")
    logging.info(f"Reduced by {100 * (start_mem - end_mem) / start_mem:.1f}%")

    return df


def get_attack_descriptions() -> dict:
    """Return descriptions of common attacks in the dataset"""
    return {
        "BENIGN": "Normal network traffic",
        "DDoS": "Distributed Denial of Service attack",
        "PortScan": "Scanning for open ports on a host",
        "Bot": "Traffic from botnet-infected devices",
        "FTP-Patator": "FTP brute force attack",
        "SSH-Patator": "SSH brute force attack",
        "DoS Hulk": "Denial of Service attack using Hulk tool",
        "DoS GoldenEye": "Denial of Service attack using GoldenEye tool",
        "DoS slowloris": "Slowloris DoS attack",
        "DoS Slowhttptest": "Slow HTTP DoS attack",
        "Heartbleed": "Heartbleed vulnerability exploit",
        "Web Attack": "Web application attack (XSS, SQL Injection, etc.)",
        "Infiltration": "Network infiltration attack"
    }
