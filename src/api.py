from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from performance_tracking import PerformanceTracker
from datetime import datetime, timedelta

app = Flask(__name__, static_folder='static')

# Load the trained models
send_time_model = joblib.load('models/send_time_model.joblib')
subject_line_model = joblib.load('models/subject_line_model.joblib')
subject_line_vectorizer = joblib.load('models/subject_line_vectorizer.joblib')

# Initialize the performance tracker
tracker = PerformanceTracker()

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict_send_time', methods=['POST'])
def predict_send_time():
    data = request.json
    hour = data.get('hour')
    day = data.get('day')
    
    if hour is None or day is None:
        return jsonify({'error': 'Missing hour or day parameter'}), 400
    
    prediction = send_time_model.predict([[hour, day]])
    predicted_opens, predicted_clicks = float(prediction[0][0]), float(prediction[0][1])
    
    # Log the prediction
    tracker.log_prediction('send_time', f'hour: {hour}, day: {day}', predicted_opens, predicted_clicks)
    
    return jsonify({
        'predicted_opens': predicted_opens,
        'predicted_clicks': predicted_clicks
    })

@app.route('/recommend_subject_line', methods=['POST'])
def recommend_subject_line():
    data = request.json
    subject = data.get('subject')
    
    if subject is None:
        return jsonify({'error': 'Missing subject parameter'}), 400
    
    vectorized_subject = subject_line_vectorizer.transform([subject])
    prediction = subject_line_model.predict(vectorized_subject)
    predicted_opens, predicted_clicks = float(prediction[0][0]), float(prediction[0][1])
    
    # Log the prediction
    tracker.log_prediction('subject_line', subject, predicted_opens, predicted_clicks)
    
    return jsonify({
        'predicted_opens': predicted_opens,
        'predicted_clicks': predicted_clicks
    })

@app.route('/log_actual_results', methods=['POST'])
def log_actual_results():
    data = request.json
    timestamp = data.get('timestamp')
    actual_opens = data.get('actual_opens')
    actual_clicks = data.get('actual_clicks')
    
    if not all([timestamp, actual_opens, actual_clicks]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    tracker.log_actual_results(timestamp, actual_opens, actual_clicks)
    return jsonify({'message': 'Actual results logged successfully'})

@app.route('/performance_metrics', methods=['GET'])
def get_performance_metrics():
    metrics = tracker.get_performance_metrics()
    return jsonify(metrics)

@app.route('/prediction_history', methods=['GET'])
def get_prediction_history():
    df = pd.read_csv('data/prediction_log.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Get the last 50 predictions, including those without actual results
    df = df.sort_values('timestamp', ascending=False).head(50)
    
    return jsonify({
        'timestamps': df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'predicted_opens': df['predicted_opens'].tolist(),
        'predicted_clicks': df['predicted_clicks'].tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)