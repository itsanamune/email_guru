# Email Optimization System

This project aims to optimize email campaigns by predicting the best send times and recommending effective subject lines.

## Setup

1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate sample email campaign data:
   ```bash
   python src/generate_sample_data.py
   ```
   This will create a file `data/email_campaign_data.csv` with simulated email campaign data.

## Data Preprocessing and Model Training

To preprocess the data, generate simulated data, and train the models, run:

```bash
python src/data_preprocessing.py
python src/data_simulation.py
python src/ml_models.py
```

This will create:
- `data/preprocessed_data.csv` with the cleaned and feature-extracted data
- `simulated_data/simulated_email_data.csv` with the simulated data
- Model files in the `models/` directory:
  - `send_time_model.joblib`: Model for predicting optimal send times
  - `subject_line_model.joblib`: Model for recommending subject lines
  - `subject_line_vectorizer.joblib`: TF-IDF vectorizer for processing subject lines

## Running the Application

To start the Flask application, run:

```bash
python src/api.py
```

The application will be available at `http://localhost:5000`. Open this URL in your web browser to access the user interface.

## Using the User Interface

1. Enter the hour (0-23) and day of the week (0-6, where 0 is Monday) for the send time prediction.
2. Enter a subject line for the subject line recommendation.
3. Click the "Predict" button to see the results.

The interface will display predicted opens and clicks for both the send time and subject line.

## API Endpoints

The following API endpoints are available:

- POST `/predict_send_time`: Predict optimal send time
  - Input: JSON with `hour` and `day` fields
  - Output: Predicted opens and clicks

- POST `/recommend_subject_line`: Recommend subject line
  - Input: JSON with `subject` field
  - Output: Predicted opens and clicks

## Testing the System

After setting up and before moving on to further development:

1. Ensure all requirements are installed: `pip install -r requirements.txt`
2. Run the data preprocessing and model training scripts as described above.
3. Start the Flask application: `python src/api.py`
4. Open `http://localhost:5000` in a web browser and test the UI with sample inputs.

If you encounter any issues during testing, please check the console output for error messages and ensure all files are in their correct locations.

## Next Steps

- Set up performance tracking and metrics visualization.
- Deploy the system to a cloud platform.