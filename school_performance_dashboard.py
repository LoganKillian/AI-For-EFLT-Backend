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

# Initialize saved_models dictionary, where users can save and later load models
saved_models = {}

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
        first_columns = ['leaid', 'leanm', 'grade', 'year', 'achv', 'achvz']
        other_columns = [col for col in g.df.columns if col not in first_columns]
        g.df = g.df[first_columns + other_columns]
    return g.df

def get_df_sub():
    """
    Retrieves and preprocesses a subset of the DataFrame.

    This function creates a subset of the main DataFrame by dropping specific columns 
    and handling missing values and non-numeric columns. The 'leanm' and 'Locale4' columns 
    are retained even if they are non-numeric. The subset is cached in the Flask `g` object.

    Returns:
        pd.DataFrame: The preprocessed subset of the main DataFrame.
    """
    if 'df_sub' not in g:
        df = get_df()
        g.df_sub = df.copy()
        
        # Columns to drop based on codebook
        ignore_cols = [
            'leaid', 'achv',
            'LOCALE_VARS', 'DIST_FACTORS', 'COUNTY_FACTORS', 'HEALTH_FACTORS',
            'CT_EconType', 'BlackBeltSm', 'Locale3'
        ]
        
        # Keep identifier columns
        id_cols = ['year', 'grade', 'leanm', 'Locale4']
        
        # Drop specified columns but keep identifier columns
        g.df_sub.drop(columns=[col for col in ignore_cols if col in g.df_sub.columns], 
                     inplace=True)
        
        # Handle true categorical features with one-hot encoding
        categorical_cols = [
            'FoodDesert',
            'CT_LowEducation', 
            'CT_PopLoss', 
            'CT_RetireDest',
            'CT_PersistPoverty', 
            'CT_PersistChildPoverty'
        ]  # Removed 'Locale4' from this list
        
        # First handle missing values in categorical columns
        for col in categorical_cols:
            if col in g.df_sub.columns:
                # Fill missing values with mode (most frequent value)
                mode_value = g.df_sub[col].mode()[0]
                g.df_sub[col].fillna(mode_value, inplace=True)
                # Create dummy variables and drop original column
                dummies = pd.get_dummies(g.df_sub[col], prefix=col, dummy_na=False)
                g.df_sub = pd.concat([g.df_sub, dummies], axis=1)
                g.df_sub.drop(columns=[col], inplace=True)
        
        # Handle numeric columns
        numeric_cols = [col for col in g.df_sub.columns 
                       if col not in id_cols and 
                       g.df_sub[col].dtype in ['int64', 'float64']]
        
        # Fill missing values with mean for numeric columns
        for column in numeric_cols:
            if g.df_sub[column].isnull().any():
                mean = g.df_sub[column].mean()
                g.df_sub.fillna({column: mean}, inplace=True)
        
        # Convert any remaining numeric columns that might be stored as objects
        for column in g.df_sub.columns:
            if (column not in id_cols and 
                column not in categorical_cols and 
                g.df_sub[column].dtype == 'object'):
                g.df_sub[column] = pd.to_numeric(g.df_sub[column], errors='coerce')
                if g.df_sub[column].isnull().any():
                    mean = g.df_sub[column].mean()
                    g.df_sub.fillna({column: mean}, inplace=True)
    
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

@app.route('/api/districts', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
@limiter.limit("10 per minute")
def get_districts():
    """
    Fetches all district names.

    Returns:
        Response: A JSON response containing the list of district names.
    """
    df = get_df()
    districts = df['leanm'].unique().tolist()
    return jsonify({"districts": districts})

@app.route('/api/grades', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
@limiter.limit("10 per minute")
def get_grades():
    """
    Fetches all grade levels.

    Returns:
        Response: A JSON response containing the list of grade levels.
    """
    df = get_df()
    grades = df['grade'].unique().tolist()
    return jsonify({"grades": grades})


@app.route('/api/years', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
@limiter.limit("10 per minute")
def get_years():
    """
    Fetches all years.

    Returns:
        Response: A JSON response containing the list of years.
    """
    df = get_df()
    years = df['year'].unique().tolist()
    return jsonify({"years": years})

@app.route('/api/locales', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
@limiter.limit("10 per minute")
def get_locales():
    """
    Fetches all locales.

    Returns:
        Response: A JSON response containing the list of locales.
    """
    df = get_df()
    locales = df['Locale4'].dropna().unique().tolist()
    locales.sort()
    return jsonify({"locales": locales})

# @app.route('api/high_percentage_demographics', methods=['GET'])
# @cache.cached(timeout=300, query_string=True)
# @limiter.limit("10 per minute")
# def get_high_percentage_demographics():
#     """
#     Fetches all high percentage demographics (Hpa)

#     Returns:
#         Response: A JSON response containing the list of hpas
#     """
#     df = get_df()
#     hpas = ["Hperasn", "Hperblk", "Hperhsp", "Hperind", "Hperwht", "Hperecd", "Hperell"]
#     return ?
    
@app.route('/api/filter_data', methods=['GET'])
@limiter.limit("10 per minute")
def filter_data():
    """
    Filters data based on query parameters and returns a paginated response.

    Args:
        district_name (str): Comma-separated list of district names to filter by.
        locale (str): Comma-separated list of locales to filter by.
        year (str): Comma-separated list of years to filter by.

    Returns:
        Response: A JSON response with filtered data and pagination details.
    """
    # Get filter parameters
    district_names = request.args.get('district_name', '').split(',')
    locales = request.args.get('locale', '').split(',')
    years = request.args.get('year', '').split(',')
    
    if years != ['all'] and years != ['']:
        years = [int(year) for year in years if year.strip()]

    df = get_df()
    
    # Apply filters
    if district_names and district_names != ['all']:
        df = df[df['leanm'].isin(district_names)]
    
    if locales and locales != ['all']:
        df = df[df['Locale4'].isin(locales)]
        
    if years and years != ['all'] and years != ['']:
        df = df[df['year'].isin(years)]

    placeholder = ""
    df = df.where(pd.notna(df), placeholder)
    
    # Convert to list of dicts to preserve column order
    data_json = df.to_dict(orient='records')

    response = {
        'data': data_json,
        'columns': df.columns.tolist(),
        'total_rows': len(df)
    }

    return jsonify(response)

@app.route('/api/get_features', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
@limiter.limit("50 per minute")
def get_features():
    """
    Retrieves the feature names from the subset DataFrame
    
    Returns:
        Response: A JSON response containing a list of feature names.
    """

    df_sub = get_df_sub()
    
    # Identifiers and target variables to exclude
    exclude_ids = ['leaid', 'leanm', 'year', 'grade', 'math', 'rla', 'achvz']

    features = [
        col for col in df_sub.columns 
        if (col not in exclude_ids) 
            #and df_sub[col].dtype in ['int64', 'float64'])
    ]

    return jsonify(features)

# TODO: Decide on if some or all categorical variables are sent with this route (used for feature selection drop down box)
@app.route('/api/get_tunable_features', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
@limiter.limit("50 per minute")
def get_tunable_features():
    """
    Retrieves the tunable feature names from the subset DataFrame, excluding:
    - Categorical variables
    - Identifier columns (leaid, leanm, year, grade)
    - Target variable (achvz)
    - Ignored columns
    
    Returns:
        Response: A JSON response containing a list of continuous feature names suitable for tuning.
    """
    df_sub = get_df_sub()
    
    # Identifiers and target variables to exclude
    exclude_ids = ['leaid', 'leanm', 'year', 'grade', 'math', 'rla', 'achvz']
    
    # Categorical variables to exclude for
    categorical_vars = [
        'Locale4', 'FoodDesert', 'CT_LowEducation', 'CT_PopLoss', 
        'CT_RetireDest', 'CT_PersistPoverty', 'CT_PersistChildPoverty',
        'Locale4_Rural', 'Local4_Suburb', 'Locale4_Town', 'Locale4_Urban',
        'FoodDesert_0.0', 'FoodDesert_1.0', 'CT_LowEducation_0.0',
        'CT_LowEducation_1.0', 'CT_PopLoss_0.0', 'CT_PopLoss_1.0',
        'CT_RetireDest_0.0', 'CT_RetireDest_1.0', 'CT_PersistPoverty_0.0',
        'CT_PersistPoverty_1.0', 'CT_PersistChildPoverty_0.0', 'CT_PersistChildPoverty_1.0'
    ]

    fixed_vars = [
        'perasn', 'perblk', 'perhsp', 'perind', 'perwht', 'perecd' , 'perell', 'CmPoverty',
        'CmUnemp', 'CmSglPrnt', 'CmCollege', 'snapall'
    ]
    
    # Get all columns that are not in exclusion lists
    features = [
        col for col in df_sub.columns 
        if (col not in exclude_ids and 
            col not in categorical_vars and 
            col not in fixed_vars)
            # and df_sub[col].dtype in ['int64', 'float64'])
    ]
    
    return jsonify(features)

# TODO: Python doc comments
@app.route('/api/run_lasso', methods=['POST'])
def run_lasso():
    """
    Run LASSO regression with filters for district, locale, and year.
    """
    tolerance = request.json.get('tolerance')
    alpha = request.json.get('alpha')
    districts = request.json.get('districts', [])
    locales = request.json.get('locales', [])
    years = request.json.get('years', [])
    
    # Debug logging
    logging.info(f"Initial filters - districts: {districts}, locales: {locales}, years: {years}")
    
    # Convert years to integers if they're not 'all'
    if years and years != ['all']:
        years = [int(year) for year in years if year.strip()]
        logging.info(f"Converted years: {years}")

    # Get the original dataframe first to apply locale filtering
    df = get_df()
    logging.info(f"Initial dataframe shape: {df.shape}")
    
    # Apply filters on the original dataframe
    if districts and districts != ['all']:
        df = df[df['leanm'].isin(districts)]
        logging.info(f"After district filter shape: {df.shape}")
    
    if locales and locales != ['all']:
        df = df[df['Locale4'].isin(locales)]
        logging.info(f"After locale filter shape: {df.shape}")
        
    if years and years != ['all']:
        df = df[df['year'].isin(years)]
        logging.info(f"After year filter shape: {df.shape}")
    
    # Log unique values in each column to verify filters
    logging.info(f"Unique districts in filtered data: {df['leanm'].unique()}")
    logging.info(f"Unique locales in filtered data: {df['Locale4'].unique()}")
    logging.info(f"Unique years in filtered data: {df['year'].unique()}")
    
    # Now get the subset with preprocessed features for the filtered rows
    df_sub = get_df_sub()
    # Filter df_sub to match the filtered rows from original df
    df_sub = df_sub[df_sub.index.isin(df.index)]
    
    df_sub = df_sub.drop(columns=['leanm'])
    
    lasso_model, metrics, feature_importance = models.lasso_cv(df_sub, tolerance, alpha)
    
    response = {
        'metrics': metrics.to_dict(),
        'feature_importance': feature_importance,
    }
    
    return jsonify(response)

@app.route('/api/adjust_features', methods=['POST'])
@limiter.limit("50 per minute")
def adjust_features():
    """
    Adjust features and run predictions with filters for district, locale, and year.
    """
    features = request.json.get('features')
    districts = request.json.get('districts', [])
    locales = request.json.get('locales', [])
    years = request.json.get('years', [])
    
    # Debug logging
    logging.info(f"Initial filters - districts: {districts}, locales: {locales}, years: {years}")
    
    # Convert years to integers if they're not 'all'
    if years and years != ['all']:
        years = [int(year) for year in years if year.strip()]
        logging.info(f"Converted years: {years}")
    
    # Get the original dataframe first to apply locale filtering
    df = get_df()
    logging.info(f"Initial dataframe shape: {df.shape}")
    
    # Apply filters on the original dataframe
    if districts and districts != ['all']:
        df = df[df['leanm'].isin(districts)]
        logging.info(f"After district filter shape: {df.shape}")
    
    if locales and locales != ['all']:
        df = df[df['Locale4'].isin(locales)]
        logging.info(f"After locale filter shape: {df.shape}")
        
    if years and years != ['all']:
        df = df[df['year'].isin(years)]
        logging.info(f"After year filter shape: {df.shape}")
    
    # Log unique values in each column to verify filters
    logging.info(f"Unique districts in filtered data: {df['leanm'].unique()}")
    logging.info(f"Unique locales in filtered data: {df['Locale4'].unique()}")
    logging.info(f"Unique years in filtered data: {df['year'].unique()}")
    
    # Now get the subset with preprocessed features for the filtered rows
    df_sub = get_df_sub()
    # Filter df_sub to match the filtered rows from original df
    df_sub = df_sub[df_sub.index.isin(df.index)]
    
    # Rest of the function remains the same
    for feature, percentage in features.items():
        if feature in df_sub.columns:
            original_mean = df_sub[feature].mean()
            df_sub[feature] *= (1 + percentage / 100)
            new_mean = df_sub[feature].mean()
            logging.info(f"Adjusted feature '{feature}' by {percentage}%")
            logging.info(f"  - Original mean: {original_mean:.2f}")
            logging.info(f"  - New mean: {new_mean:.2f}")
            logging.info(f"  - Actual change: {((new_mean - original_mean) / original_mean * 100):.2f}%")
    
    ext_model, metrics, comparison_df, feature_importance = models.ext_trees(df_sub, feature_adjustments=features)
    
    comparison_df['leanm'] = df_sub['leanm']
    comparison_df['grade'] = df_sub['grade']
    comparison_df['year'] = df_sub['year']
    
    summary_stats = {
        'mean_original_achvz': float(comparison_df['original_achvz'].mean()),
        'mean_predicted_achvz': float(comparison_df['predicted_achvz'].mean()),
        'change_in_achvz': float(comparison_df['predicted_achvz'].mean() - comparison_df['original_achvz'].mean())
    }
    
    for feature in features.keys():
        orig_col = f'original_{feature}'
        adj_col = f'adjusted_{feature}'
        summary_stats[f'mean_{orig_col}'] = float(comparison_df[orig_col].mean())
        summary_stats[f'mean_{adj_col}'] = float(comparison_df[adj_col].mean())
        summary_stats[f'change_in_{feature}'] = float(comparison_df[adj_col].mean() - comparison_df[orig_col].mean())
    
    response = {
        'comparison_data': comparison_df.to_dict(orient='records'),
        'summary_stats': summary_stats,
        'metrics': metrics.to_dict(),
        'feature_importance': feature_importance
    }
    
    return jsonify(response)

# TODO: new API route for adjusting target values and finding new feature values
#@app.route('/api/adjust_target', methods=['POST'])

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

@app.route('/api/get_feature_descriptions', methods=['GET'])
def get_feature_descriptions():
    """
    Returns a dictionary of feature descriptions.
    """
    feature_descriptions = {
        "leaid": "NCES ID - Geographic School Districts",
        "leanm": "District Name",
        "grade": "Tested Grade (g)",
        "year": "Spring of Tested Year (y)",
        "achv": "Di Math + RLA Achievement",
        "achvz": "Year STD Achv",
        "math": "Di Mth Achvment",
        "rla": "Di Read/LA Achvment",
        "totenrl": "Number of Students in Grade",
        "perasn": "Percent Asians in the grade",
        "perblk": "Percent Blacks in the grade",
        "perhsp": "Percent Hispanics in the grade",
        "perind": "Percent Native Americans in the grade",
        "perwht": "Percent Whites in the grade",
        "perecd": "Percent Economic Disadvantage in the grade",
        "perell": "Percent English Language Learners in district",
        "Hperasn": "1=High Percentage (>50%) Asian",
        "Hperblk": "1=High Percentage (>50%) Black",
        "Hperhsp": "1=High Percentage (>50%) Hispanic",
        "Hperind": "1=High Percentage (>50%) Am Indian/AK Native",
        "Hperwht": "1=High Percentage (>50%) White",
        "Hperecd": "1=High Percentage (>50%) Econ Disadvantage",
        "Hperell": "1=High Percentage (>50%) English Language Learner",
        "LOCALE_VARS": "----Locale & Region----",
        "Locale4": "Locale: 1Urb 2Sub 3Town 4Rural",
        "Locale3": "1S&T 2U 3R",
        "BlackBeltSm": "Black Belt Counties",
        "DIST_FACTORS": "----District Characteristics----",
        "DiSTR": "Di St Tchr Ratio",
        "DiInstAid": "Inst Aides per Student",
        "DiGuidC": "Counselors per Student",
        "DiLib": "Librarians per Student",
        "DiAdmin": "Administrators per Student",
        "DiRevTotal": "Total Revenue/Student",
        "DiRevLocal": "Local Revenue/Student",
        "DiExpTotal": "Total Expend/Student",
        "DiExpInst": "Instruct Spend/Student",
        "DiExpSuppt": "Support Svc. Spend/Student",
        "DiExpPupl": "Pupil Support Svc/Student",
        "DiExpPlant": "Building Oper/Student",
        "DiExpBus": "Transport/Student",
        "DiSalAll": "Total Salaries/Student",
        "DiSalReg": "Reg Tchr Sal/Student",
        "DiSalInst": "Instr Sal/Student",
        "DiSalPlant": "Maintenance Sal/Student",
        "DiSalBus": "Transport Sal/Student",
        "DiSalPupl": "Pupil Support Sal/Student",
        "hsecdnec": "Difference between ECD/Non-ECD",
        "rsecdnec": "Diversity index ECD/Non ECD",
        "COUNTY_FACTORS": "----County Characteristics----",
        "CmPoverty": "Community Poverty",
        "CmUnemp": "Community Unemployment",
        "CmSglPrnt": "Community Single Parent Households",
        "CmCollege": "Community College Attendance",
        "snapall": "SNAP receipt proportion",
        "FoodDesert": "Food Desert",
        "FoodDesert_0.0": "Food Desert",
        "FoodDesert_1.0": "Food Desert",
        "CT_EconType": "Dominant Industry in Area",
        "CT_LowEducation": "Low Education County",
        "CT_PopLoss": "Population Decline",
        "CT_RetireDest": "Retirement Area",
        "CT_PersistPoverty": "Generational Poverty",
        "CT_PersistChildPoverty": "Generational Child Poverty",
        "CT_EconType_0.0": "Dominant Industry in Area",
        "CT_LowEducation_0.0": "Low Education County",
        "CT_PopLoss_0.0": "Population Decline",
        "CT_RetireDest_0.0": "Retirement Area",
        "CT_PersistPoverty_0.0": "Generational Poverty",
        "CT_PersistChildPoverty_0.0": "Generational Child Poverty",
        "CT_EconType_1.0": "Dominant Industry in Area",
        "CT_LowEducation_1.0": "Low Education County",
        "CT_PopLoss_1.0": "Population Decline",
        "CT_RetireDest_1.0": "Retirement Area",
        "CT_PersistPoverty_1.0": "Generational Poverty",
        "CT_PersistChildPoverty_1.0": "Generational Child Poverty",
        "HEALTH_FACTORS": "----County Health Factors Scale----",
        "H_FCTR_ZS": "County Health Scale: All Factors",
        "HBehaveZS": "CHS Health Behaviors",
        "ClinCareZS": "CHS Clinical Care Access",
        "SocEcnZS": "CHS Soc/Econ Environment",
        "PhysEnvZS": "CHS Physical Environment"
    }
    return jsonify(feature_descriptions)
if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT", default=5000))
