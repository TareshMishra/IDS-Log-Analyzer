import sys
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import PySimpleGUI as sg
from backend.core.train import TrainingPipeline
from backend.core.predict import PredictionPipeline
from backend.core.config import app_config, model_config

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IDSAnalyzerGUI:
    def __init__(self):
        self.pipeline = PredictionPipeline()
        self.current_file = None
        self.results = None

        # Initialize GUI theme
        try:
            sg.theme(app_config.GUI_THEME)
        except Exception as e:
            logger.warning(f"Could not set GUI theme: {str(e)}")
            sg.SetOptions(
                background_color='#2B2B2B',
                text_element_background_color='#2B2B2B',
                element_background_color='#2B2B2B',
                text_color='#FFFFFF',
                input_text_color='#FFFFFF',
                button_color=('white', '#475841')
            )

    def create_window(self):
        """Create the main application window"""
        layout = [
            [sg.Text("CIC IDS Log Analyzer", font=('Helvetica', 20)),
             sg.Button('Exit', size=(5, 1), pad=((400, 0), (0, 0)))],
            [sg.HorizontalSeparator()],
            [
                sg.Column([
                    [sg.Text("Dataset File:")],
                    [sg.Input(key='-FILE-', size=(40, 1)), sg.FileBrowse()],
                    [sg.Button('Load', key='-LOAD-'),
                     sg.Button('Analyze', key='-ANALYZE-', disabled=True),
                     sg.Button('Train', key='-TRAIN-')],
                    [sg.HorizontalSeparator()],
                    [sg.Text("Results Summary:", font=('Helvetica', 14))],
                    [sg.Multiline(key='-RESULTS-', size=(55, 10),
                     font=('Courier', 10), disabled=True)],
                    [sg.Button('Details', key='-STATS-', disabled=True),
                     sg.Button('Export', key='-EXPORT-', disabled=True)]
                ], pad=(10, 10)),

                sg.Column([
                    [sg.Canvas(key='-CANVAS-', size=(400, 300))],
                    [sg.Text("Attack Distribution:", font=('Helvetica', 14))],
                    [sg.Image(key='-IMAGE-')]
                ], pad=(10, 10))
            ],
            [sg.StatusBar("", key='-STATUS-', size=(60, 1),
                          relief=sg.RELIEF_SUNKEN)]
        ]

        return sg.Window("CIC IDS Analyzer", layout, finalize=True, resizable=True)

    def run(self):
        """Main event loop for the GUI"""
        window = self.create_window()

        while True:
            event, values = window.read(timeout=100)

            if event in (sg.WINDOW_CLOSED, 'Exit'):
                break

            elif event == '-LOAD-':
                self.handle_load_event(window, values)

            elif event == '-ANALYZE-':
                self.handle_analyze_event(window)

            elif event == '-TRAIN-':
                self.handle_train_event(window)

            elif event == '-ANALYSIS_DONE-':
                self.handle_analysis_done(window, values)

            elif event == '-TRAINING_DONE-':
                self.handle_training_done(window, values)

            elif event == '-STATS-':
                self.handle_stats_event(window)

            elif event == '-EXPORT-':
                self.handle_export_event(window)

        window.close()

    def handle_load_event(self, window, values):
        """Handle file loading"""
        file_path = values['-FILE-']
        if file_path and Path(file_path).is_file():
            try:
                self.current_file = file_path
                window['-ANALYZE-'].update(disabled=False)
                window['-STATUS-'].update(f"Loaded: {Path(file_path).name}")
            except Exception as e:
                window['-STATUS-'].update(f"Load error: {str(e)}")
                sg.popup_error(f"Error loading file:\n{str(e)}")

    def handle_analyze_event(self, window):
        """Handle analysis start"""
        if not self.current_file:
            sg.popup_error("No file loaded!", title="Error")
            return

        model_path = Path(model_config.MODEL_DIR) / \
            model_config.SUPERVISED_MODEL_NAME
        if not model_path.exists():
            if sg.popup_yes_no("Models not found. Train first?", title="Warning") == "Yes":
                self.handle_train_event(window)
            return

        window['-STATUS-'].update("Analyzing...")
        window['-RESULTS-'].update("")
        window['-STATS-'].update(disabled=True)
        window['-EXPORT-'].update(disabled=True)

        try:
            window.perform_long_operation(
                lambda: self.pipeline.analyze_file(self.current_file),
                '-ANALYSIS_DONE-'
            )
        except Exception as e:
            window['-STATUS-'].update(f"Analysis failed: {str(e)}")
            sg.popup_error(f"Could not start analysis:\n{str(e)}")

    def handle_train_event(self, window):
        """Handle model training"""
        window['-STATUS-'].update("Training models...")
        window['-RESULTS-'].update("Training in progress...")
        window['-STATS-'].update(disabled=True)
        window['-EXPORT-'].update(disabled=True)

        try:
            window.perform_long_operation(
                lambda: TrainingPipeline().run(),
                '-TRAINING_DONE-'
            )
        except Exception as e:
            window['-STATUS-'].update(f"Training failed: {str(e)}")
            sg.popup_error(f"Could not start training:\n{str(e)}")

    def handle_analysis_done(self, window, values):
        """Handle completed analysis"""
        try:
            result = values['-ANALYSIS_DONE-']

            # Check for error response
            if isinstance(result, dict) and 'error' in result:
                raise Exception(result['error'])

            self.results = result

            # Validate results structure
            if not isinstance(self.results, dict) or 'results' not in self.results:
                raise ValueError("Invalid results format received")

            # Convert to DataFrame if needed
            if not isinstance(self.results['results'], pd.DataFrame):
                self.results['results'] = pd.DataFrame(self.results['results'])

            # Generate statistics
            stats = self.pipeline.generate_attack_stats(
                self.results['results'])

            # Build results text
            result_text = f"Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            result_text += f"Total Records: {len(self.results['results']):,}\n"

            if 'anomaly_rate' in stats:
                result_text += f"Anomaly Rate: {stats['anomaly_rate']:.2f}%\n"
            if 'avg_anomaly_score' in stats:
                result_text += f"Avg Anomaly Score: {stats['avg_anomaly_score']:.4f}\n"
            if 'accuracy' in stats:
                result_text += f"Accuracy: {stats['accuracy']:.2f}%\n"

            # Update GUI
            window['-RESULTS-'].update(result_text)
            window['-STATS-'].update(disabled=False)
            window['-EXPORT-'].update(disabled=False)
            window['-STATUS-'].update("Analysis completed")

            # Update chart if we have attack distribution
            if 'attack_distribution' in stats and stats['attack_distribution']:
                self.update_attack_plot(window, stats['attack_distribution'])
            else:
                window['-IMAGE-'].update(data=None)
                logger.warning("No attack distribution data available")

        except Exception as e:
            error_msg = f"Could not display results: {str(e)}"
            logger.error(error_msg)
            window['-STATUS-'].update(error_msg)
            window['-RESULTS-'].update(error_msg)
            sg.popup_error(error_msg, title="Display Error")

    def handle_training_done(self, window, values):
        """Handle completed training"""
        result = values['-TRAINING_DONE-']

        if isinstance(result, dict) and result.get('status') == 'success':
            window['-STATUS-'].update("Training completed")
            window['-RESULTS-'].update("Model training completed successfully!\n" +
                                       f"Models saved to: {result.get('model_path', '')}")
            sg.popup("Model training completed!", title="Success")
        else:
            error_msg = result.get('message', 'Unknown training error')
            window['-STATUS-'].update(f"Training failed: {error_msg}")
            window['-RESULTS-'].update(f"Training error: {error_msg}")
            sg.popup_error(f"Training failed:\n{error_msg}", title="Error")

    def handle_stats_event(self, window):
        """Show detailed statistics"""
        if not self.results or 'results' not in self.results:
            sg.popup_error("No results available!", title="Error")
            return

        try:
            stats = self.pipeline.generate_attack_stats(
                self.results['results'])
            stats_text = "Detailed Statistics:\n\n"

            if 'attack_distribution' in stats:
                stats_text += "Attack Distribution:\n"
                for attack, count in stats['attack_distribution'].items():
                    stats_text += f"  {attack}: {count:,}\n"
                stats_text += "\n"

            if 'anomaly_rate' in stats:
                stats_text += f"Anomaly Rate: {stats['anomaly_rate']:.2f}%\n"
            if 'avg_anomaly_score' in stats:
                stats_text += f"Average Anomaly Score: {stats['avg_anomaly_score']:.4f}\n"
            if 'accuracy' in stats:
                stats_text += f"Classification Accuracy: {stats['accuracy']:.2f}%\n"
            if 'total_records' in stats:
                stats_text += f"\nTotal Records Processed: {stats['total_records']:,}"

            sg.popup_scrolled(
                stats_text, title="Detailed Statistics", size=(50, 15))
        except Exception as e:
            sg.popup_error(f"Could not show details:\n{str(e)}", title="Error")

    def handle_export_event(self, window):
        """Export results to CSV"""
        if not self.results or 'results' not in self.results:
            sg.popup_error("No results to export!", title="Error")
            return

        try:
            save_path = sg.popup_get_file(
                "Save results as CSV",
                save_as=True,
                default_extension=".csv",
                file_types=(("CSV Files", "*.csv"),)
            )

            if save_path:
                self.results['results'].to_csv(save_path, index=False)
                sg.popup(
                    f"Results saved to:\n{save_path}", title="Export Successful")
        except Exception as e:
            sg.popup_error(f"Export failed:\n{str(e)}", title="Error")

    def update_attack_plot(self, window, attack_distribution):
        """Update the attack distribution chart"""
        try:
            import matplotlib.pyplot as plt
            from io import BytesIO

            # Prepare data
            attacks = list(attack_distribution.keys())
            counts = list(attack_distribution.values())

            # Create figure
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(attacks, counts)

            # Customize plot
            ax.set_title('Attack Distribution', pad=20)
            ax.set_ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Save to buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)

            # Update GUI
            window['-IMAGE-'].update(data=buf.read())
            buf.close()
        except Exception as e:
            logger.error(f"Chart update failed: {str(e)}")
            window['-IMAGE-'].update(data=None)


def run_gui():
    """Run the GUI application"""
    try:
        app = IDSAnalyzerGUI()
        app.run()
    except Exception as e:
        logger.error(f"GUI failed: {str(e)}")
        sg.popup_error(f"Fatal error:\n{str(e)}")
        sys.exit(1)
