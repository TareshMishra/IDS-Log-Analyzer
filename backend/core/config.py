from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    RAW_DATA_DIR: str = field(default=str(
        Path(__file__).parent.parent / "C:\\Users\\asus\\Desktop\\V222222\\ids_log_analysis\\data\\CICIDS2017_improved"))
    PROCESSED_DATA_DIR: str = field(default=str(
        Path(__file__).parent.parent / "C:\\Users\\asus\\Desktop\\V222222\\ids_log_analysis\\data\\processed_data"))
    TRAIN_FILE: str = field(default="monday.csv")
    TEST_FILES: list = field(default_factory=list)
    LABEL_COLUMN: str = field(default="Label")


@dataclass
class ModelConfig:
    MODEL_DIR: str = field(default=str(
        Path(__file__).parent.parent / "backend/models"))
    SUPERVISED_MODEL_NAME: str = field(default="random_forest.pkl")
    UNSUPERVISED_MODEL_NAME: str = field(default="isolation_forest.pkl")
    SCALER_NAME: str = field(default="scaler.pkl")
    RANDOM_STATE: int = field(default=42)
    TEST_SIZE: float = field(default=0.2)
    ANOMALY_THRESHOLD: float = field(default=-0.2)
    BENIGN_LABEL: str = field(default="BENIGN")


@dataclass
class AppConfig:
    DEBUG: bool = field(default=True)
    WEB_PORT: int = field(default=5000)
    GUI_THEME: str = field(default="DarkAmber")
    MAX_FILE_SIZE_MB: int = field(default=4096)


# Initialize configurations
config = DataConfig()
model_config = ModelConfig()
app_config = AppConfig()
