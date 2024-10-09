import pandas as pd
from datetime import datetime

class PerformanceTracker:
    def __init__(self, log_file='data/prediction_log.csv'):
        self.log_file = log_file
        self.columns = ['timestamp', 'prediction_type', 'input', 'predicted_opens', 'predicted_clicks', 'actual_opens', 'actual_clicks']
        self._initialize_log_file()

    def _initialize_log_file(self):
        try:
            pd.read_csv(self.log_file)
        except FileNotFoundError:
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.log_file, index=False)

    def log_prediction(self, prediction_type, input_data, predicted_opens, predicted_clicks):
        log_entry = {
            'timestamp': datetime.now(),
            'prediction_type': prediction_type,
            'input': str(input_data),
            'predicted_opens': predicted_opens,
            'predicted_clicks': predicted_clicks,
            'actual_opens': None,
            'actual_clicks': None
        }
        df = pd.DataFrame([log_entry])
        df.to_csv(self.log_file, mode='a', header=False, index=False)

    def log_actual_results(self, timestamp, actual_opens, actual_clicks):
        df = pd.read_csv(self.log_file)
        mask = (df['timestamp'] == timestamp) & (df['actual_opens'].isnull())
        df.loc[mask, 'actual_opens'] = actual_opens
        df.loc[mask, 'actual_clicks'] = actual_clicks
        df.to_csv(self.log_file, index=False)

    def get_performance_metrics(self):
        df = pd.read_csv(self.log_file)
        df = df.dropna()  # Only consider entries with actual results

        metrics = {
            'total_predictions': len(df),
            'mse_opens': float(((df['predicted_opens'] - df['actual_opens']) ** 2).mean()) if not df.empty else 0,
            'mse_clicks': float(((df['predicted_clicks'] - df['actual_clicks']) ** 2).mean()) if not df.empty else 0,
            'mae_opens': float((df['predicted_opens'] - df['actual_opens']).abs().mean()) if not df.empty else 0,
            'mae_clicks': float((df['predicted_clicks'] - df['actual_clicks']).abs().mean()) if not df.empty else 0,
        }

        # Convert NaN to 0
        metrics = {k: 0 if pd.isna(v) else v for k, v in metrics.items()}

        return metrics