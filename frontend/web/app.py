from backend.core.config import app_config, config, model_config
from backend.core.train import TrainingPipeline
from backend.core.predict import PredictionPipeline
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = app_config.MAX_FILE_SIZE_MB * 1024 * 1024

# Ensure upload directory exists
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

pipeline = PredictionPipeline()


def validate_csv(filepath):
    """Basic CSV validation"""
    try:
        df = pd.read_csv(filepath, nrows=5)
        if config.LABEL_COLUMN not in df.columns:
            raise ValueError(
                f"Required column '{config.LABEL_COLUMN}' missing")
        return True
    except Exception as e:
        logger.error(f"CSV validation failed: {str(e)}")
        raise


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files allowed'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Validate CSV structure
        validate_csv(filepath)

        results = pipeline.analyze_file(filepath)

        # Safely generate stats with defaults
        stats = pipeline.generate_attack_stats(
            results.get('results', pd.DataFrame()))

        # Build response data with fallback values
        response_data = {
            'filename': filename,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'anomaly_rate': f"{stats.get('anomaly_rate', 0):.2f}%",
            'avg_anomaly_score': f"{stats.get('avg_anomaly_score', 0):.4f}",
            'attack_distribution': stats.get('attack_distribution', {}),
            'results': results.get('results', pd.DataFrame()).to_dict(orient='records')[:100]
        }

        if 'accuracy' in stats:
            response_data['accuracy'] = f"{stats['accuracy']:.2f}%"

        return jsonify(response_data)

    except pd.errors.EmptyDataError:
        return jsonify({'error': 'The CSV file is empty'}), 400
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Analysis failed: {traceback.format_exc()}")
        return jsonify({
            'error': 'Analysis failed',
            'details': str(e),
            'trace': traceback.format_exc() if app_config.DEBUG else None
        }), 500


@app.route('/train', methods=['POST'])
def train_model():
    try:
        train_pipeline = TrainingPipeline()
        results = train_pipeline.run()

        # Reset prediction pipeline after training
        global pipeline
        pipeline = PredictionPipeline()

        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully',
            'model_info': results.get('model_info', {})
        })
    except Exception as e:
        logger.error(f"Training failed: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'trace': traceback.format_exc() if app_config.DEBUG else None
        }), 500


@app.route('/download_sample', methods=['GET'])
def download_sample():
    try:
        sample_path = Path(config.RAW_DATA_DIR) / config.TRAIN_FILE
        if not sample_path.exists():
            raise FileNotFoundError("Sample file not found")

        return send_file(
            sample_path,
            as_attachment=True,
            download_name='sample_network_traffic.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 404


if __name__ == '__main__':
    app.run(debug=app_config.DEBUG, port=app_config.WEB_PORT)
