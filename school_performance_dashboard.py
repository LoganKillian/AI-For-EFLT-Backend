import os
from flask import Flask, jsonify, request, Response, g
from flask_cors import CORS
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler, normalize
import tqdm
import queue
import threading
import logging
import io
import json

from feature_predictor import FeaturePredictor
import models

app = Flask(__name__)
CORS(app)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Setup rate limiter
limiter = Limiter(get_remote_address, app=app)

# Setup log queue, stream, and format, and get loggers from other files
log_queue = queue.Queue()
log_capture_string = io.StringIO()
ch = logging.StreamHandler(log_capture_string)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logging.getLogger('').addHandler(ch)
logging.getLogger('feature_predictor').addHandler(ch)
logging.getLogger('models').addHandler(ch)

# Constant to determine minimum number of rows where application does not issue warning
# TODO: Trial and error test models with lowest amount of possible rows with still good predictions
MINIMUM_REQUIRED_ROWS = 150

# Initialize saved_models dictionary, where users can save and later load models
saved_models = {}

# Initialize locked_features set, which holds features that should not be adjusted
# TODO: add feature names to this set and make sure adjusting features does not use features in this set
locked_features = set()

def log_monitor():
    """Monitors the log stream and adds log entries to a queue."""
    while True:
        log_capture_string.seek(0)
        lines = log_capture_string.readlines()
        for line in lines:
            log_queue.put_nowait(line.strip())
        log_capture_string.truncate(0)
        log_capture_string.seek(0)

# Start log monitoring in a separate thread
threading.Thread(target=log_monitor, daemon=True).start()

def get_df():
    """
    Retrieves the main DataFrame from a CSV file.
    
    This function reads 'AL_Dist.csv' and loads the data into a Pandas DataFrame.
    The DataFrame is stored in the Flask `g` object to maintain a session-specific
    global variable.

    Returns:
        pd.DataFrame: The main DataFrame.
    """
    if 'df' not in g:
        g.df = pd.read_csv('AL_Dist.csv')
    return g.df

def get_df_sub():
    """
    Retrieves and preprocesses a subset of the DataFrame.

    This function creates a subset of the main DataFrame by dropping specific columns 
    and handling missing values and non-numeric columns. The subset is cached in the 
    Flask `g` object.

    Returns:
        pd.DataFrame: The preprocessed subset of the main DataFrame.
    """
    if 'df_sub' not in g:
        df = get_df()
        g.df_sub = df.copy()
        drop_cols = ['leaid', 'achv', 'Locale3', 'math', 'rla']
        g.df_sub.drop(columns=drop_cols, inplace=True)
        
        # Remove non-numeric columns
        for column in g.df_sub.columns:
            if not g.df_sub[column].astype(str).str.contains(r'[0-9.-]', regex=True).any():
                g.df_sub.drop(columns=column, inplace=True)

        # Fill missing values with column mean
        for column in g.df_sub.columns:
            if g.df_sub[column].isnull().any():
                mean = g.df_sub[column].mean()
                g.df_sub.fillna({column: mean}, inplace=True)
    
        # Convert object columns to numeric
        for column in g.df_sub.columns:
            if g.df_sub[column].dtype == 'object':
                g.df_sub[column] = pd.to_numeric(g.df_sub[column], errors='coerce')
    
    return g.df_sub

@app.route('/api/load_data', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
@limiter.limit("10 per minute")
def load_data():
    """
    Loads data into the application.

    This endpoint loads the main DataFrame and its subset.

    Returns:
        Response: A JSON response indicating the data load status.
    """
    cache.clear()
    get_df()
    get_df_sub()
    return jsonify("Data loaded successfully")

@app.route('/api/filter_data', methods=['GET'])
@limiter.limit("10 per minute")
def filter_data():
    """
    Filters data based on query parameters and returns a paginated response.

    Args:
        start (int): The starting index of the rows.
        limit (int): The number of rows to return.
        district_name (str): The name of the district to filter by.

    Returns:
        Response: A JSON response with filtered data and pagination details.
    """
    start = int(request.args.get('start', start=0))
    rows = int(request.args.get('limit', limit=10))
    district_name = request.args.get('district_name')

    df = get_df()
    if district_name:
        district_names = district_name.split(',')
        data_slice = df[df['leanm'].isin(district_names)]
    else:
        data_slice = df

    data_slice = data_slice.iloc[start:start + rows]
    data_slice.insert(0, 'Index', (data_slice.index + 1))
    total_rows = len(df)

    placeholder = ""
    data_slice = data_slice.where(pd.notna(data_slice), placeholder)
    data_json = data_slice.to_dict(orient='records')

    total_pages = (total_rows // rows) + (1 if total_rows % rows != 0 else 0)
    current_page = (start // rows) + 1

    response = {
        'data': data_json,
        'pagination': {
            'total_rows': total_rows,
            'total_pages': total_pages,
            'current_page': current_page,
            'rows_per_page': rows
        }
    }

    return jsonify(response)

@app.route('/api/get_features', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
@limiter.limit("50 per minute")
def get_features():
    """
    Retrieves the feature names from the subset DataFrame.

    Returns:
        Response: A JSON response containing a list of feature names.
    """
    df_sub = get_df_sub()
    features = df_sub.columns.tolist()
    return jsonify(features)

@app.route('/api/get_feature_ranges', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
@limiter.limit("10 per minute")
def get_feature_ranges():
    """
    Retrieves the minimum, maximum, and mean values for each feature.

    Returns:
        Response: A JSON response with the feature ranges.
    """
    df_sub = get_df_sub()
    ranges = {}
    for column in df_sub.columns:
        ranges[column] = {
            'min': float(df_sub[column].min()),
            'max': float(df_sub[column].max()),
            'mean': float(df_sub[column].mean())
        }
    return jsonify(ranges)

@app.route('/api/get_length', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
@limiter.limit("50 per minute")
def get_length():
    """
    Retrieves the number of rows in the main DataFrame.

    Returns:
        Response: A JSON response with the number of rows.
    """
    df = get_df()
    return jsonify(len(df))

@app.route('/api/run_lasso', methods=['POST'])
def run_lasso():
    """
    Runs a Lasso regression model on the subset DataFrame.

    Args:
        tolerance (float): The tolerance for the Lasso model.
        alpha (float): The alpha value for the Lasso model.

    Returns:
        Response: A JSON response with model metrics and feature importance.
    """
    tolerance = request.json.get('tolerance')
    alpha = request.json.get('alpha')
    df_sub = get_df_sub()

    if df_sub.shape[0] < MINIMUM_REQUIRED_ROWS:
        return jsonify({
            'warning': 'Insufficient data for accurate predictions. Consider adding more data points.',
            'data_points': df_sub.shape[0]
        }), 400
    
    lasso_model, metrics, coefficients = models.lasso_cv(df_sub, tolerance, alpha)
    
    # Sort coefficients by absolute value for feature importance
    feature_importance = coefficients.abs().sort_values(by='Coefficients', ascending=False)
    
    response = {
        'metrics': metrics.to_dict(),
        'feature_importance': feature_importance.to_dict()
    }
    
    return jsonify(response)

@app.route('/api/run_extra_trees', methods=['POST'])
def run_extra_trees():
    """
    Runs an Extra Trees model on the subset DataFrame.

    Args:
        n_estimators (int): The number of trees in the forest.

    Returns:
        Response: A JSON response with model metrics and feature importances.
    """
    n_estimators = request.json.get('n_estimators', 50)
    df_sub = get_df_sub()

    if df_sub.shape[0] < MINIMUM_REQUIRED_ROWS:
        return jsonify({
            'warning': 'Insufficient data for accurate predictions. Consider adding more data points.',
            'data_points': df_sub.shape[0]
        }), 400
    
    ext_model, metrics = models.ext_trees(df_sub, n_estimators)
    
    # Get feature importances
    feature_importances = pd.DataFrame({
        'feature': df_sub.columns,
        'importance': ext_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    response = {
        'metrics': metrics.to_dict(),
        'feature_importances': feature_importances.to_dict(orient='records')
    }
    
    return jsonify(response)

@app.route('/api/adjust_features', methods=['POST'])
@limiter.limit("50 per minute")
def adjust_features():
    """
    Adjusts the features based on input and runs the Lasso model for prediction.

    Args:
        features (dict): The features to adjust.
        target (str): The target variable for prediction.

    Returns:
        Response: A JSON response with adjusted features and the prediction result.
    """
    features = request.json.get('features')
    target = request.json.get('target')
    df_sub = get_df_sub()
    
    # Run Lasso CV to get the model and coefficients
    lasso_model, metrics, coefficients = models.lasso_cv(df_sub)
    
    predictor = FeaturePredictor(lasso_model, target)
    
    predictor.initialize_weights(coefficients, df_sub, {})
    
    adjusted_df = predictor.adjust_features(pd.DataFrame(features, index=[0]))
    
    prediction = predictor.model.predict(adjusted_df)[0]
    
    response = {
        'adjusted_features': adjusted_df.to_dict(orient='records')[0],
        'prediction': float(prediction)
    }
    
    return jsonify(response)

@app.route('/api/log_stream')
def log_stream():
    """
    Streams log entries in real-time as Server-Sent Events (SSE).

    Returns:
        Response: A stream of log entries.
    """
    def generate():
        while True:
            message = log_queue.get()
            yield f"data: {message}\n\n"
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/save_model', methods=['POST'])
@limiter.limit("50 per minute")
def save_model():
    """
    Saves the current model state under a given name.

    Args:
        model_name (str): The name to save the model under.
        model_state (dict): The state of the model to save.

    Returns:
        Response: A JSON response indicating the success or failure of the operation.
    """
    model_name = request.json.get('model_name')
    model_state = request.json.get('model_state')

    if not model_name or not model_state:
        return jsonify({"error": "Model name and state are required."}), 400

    saved_models[model_name] = model_state
    return jsonify({"message": f"Model '{model_name}' saved successfully."}), 200

@app.route('/api/load_model', methods=['POST'])
@limiter.limit("50 per minute")
def load_model():
    """
    Loads a saved model state by name.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        Response: A JSON response containing the loaded model state or an error message.
    """
    model_name = request.json.get('model_name')

    if not model_name or model_name not in saved_models:
        return jsonify({"error": "Model not found."}), 404

    model_state = saved_models[model_name]
    return jsonify({"model_state": model_state}), 200

if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT", default=5000))
