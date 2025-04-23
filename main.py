import argparse
import logging
from pathlib import Path
from backend.core.config import app_config
from backend.core.train import TrainingPipeline  # Add this import
from frontend.gui.app import run_gui
from frontend.web.app import app as web_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ids_analyzer.log'),
        logging.StreamHandler()
    ]
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="CIC IDS Log Analyzer")
    parser.add_argument('--gui', action='store_true',
                        help='Run the GUI application')
    parser.add_argument('--web', action='store_true',
                        help='Run the web application')
    parser.add_argument('--train', action='store_true',
                        help='Train machine learning models')  # Add this line
    return parser.parse_args()


def main():
    """Main entry point for the application"""
    args = parse_args()

    # Create necessary directories
    Path("backend/models").mkdir(parents=True, exist_ok=True)
    Path("storage/datasets").mkdir(parents=True, exist_ok=True)
    Path("storage/processed").mkdir(parents=True, exist_ok=True)

    if args.train:  # Add this block
        logger = logging.getLogger(__name__)
        logger.info("Starting model training...")
        result = TrainingPipeline().run()
        if result.get('status') == 'success':
            logger.info(
                f"Training successful! Models saved to: {result.get('model_path', '')}")
        else:
            logger.error(
                f"Training failed: {result.get('message', 'Unknown error')}")
    elif args.gui:
        run_gui()
    elif args.web:
        web_app.run(debug=app_config.DEBUG, port=app_config.WEB_PORT)
    else:
        print("Please specify one of: --train, --gui, or --web")


if __name__ == '__main__':
    main()
