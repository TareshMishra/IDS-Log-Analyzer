import logging
import time
from typing import Dict, Any
from pathlib import Path
from .config import config, model_config
from .load_data import load_dataset
from .predict import PredictionPipeline
from .utils import timing

logger = logging.getLogger(__name__)


class IDSAgent:
    def __init__(self):
        self.pipeline = PredictionPipeline()
        self.last_processed = None
        self.running = False

    @timing
    def start_monitoring(self, directory: str = None) -> None:
        """Start monitoring a directory for new files"""
        self.running = True
        watch_dir = directory or config.RAW_DATA_DIR
        logger.info(f"Starting monitoring on directory: {watch_dir}")

        try:
            while self.running:
                self.process_new_files(watch_dir)
                time.sleep(10)  # Check every 10 seconds
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring error: {str(e)}")

    @timing
    def process_new_files(self, directory: str) -> Dict[str, Any]:
        """Process any new files in the directory"""
        results = {}
        path = Path(directory)

        for file_path in path.glob('*.csv'):
            if self.last_processed is None or file_path.stat().st_mtime > self.last_processed:
                try:
                    logger.info(f"Processing new file: {file_path.name}")
                    result = self.pipeline.analyze_file(str(file_path))
                    results[file_path.name] = result
                    self.last_processed = file_path.stat().st_mtime

                    # Generate alerts for anomalies
                    self.generate_alerts(result, file_path.name)

                except Exception as e:
                    logger.error(
                        f"Error processing {file_path.name}: {str(e)}")

        return results

    @timing
    def generate_alerts(self, analysis_result: Dict[str, Any], filename: str) -> None:
        """Generate alerts based on analysis results"""
        stats = self.pipeline.generate_attack_stats(analysis_result['results'])

        # Check for high anomaly rate
        if stats.get('anomaly_rate', 0) > 20:  # 20% threshold
            logger.warning(
                f"HIGH ANOMALY RATE in {filename}: {stats['anomaly_rate']}%")

        # Check for specific attacks
        attack_dist = stats.get('attack_distribution', {})
        for attack, count in attack_dist.items():
            if attack != config.BENIGN_LABEL and count > 10:  # More than 10 occurrences
                logger.warning(
                    f"ATTACK DETECTED in {filename}: {attack} ({count} occurrences)")

    @timing
    def stop_monitoring(self) -> None:
        """Stop the monitoring agent"""
        self.running = False
        logger.info("Monitoring agent stopped")
