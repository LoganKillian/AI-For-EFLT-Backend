import pytest
import json
from flask import Flask
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from school_performance_dashboard import app, get_df, get_df_sub
import models

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_df():
    data = {
        # Create 200 rows of data, minimum required rows for running models currently set to 150
        'leaid': np.random.choice([100005, 100006], size=200),
        'leanm': np.random.choice(['ALBERTVILLE CITY', 'MARSHALL COUNTY', 'HOOVER CITY'], size=200),
        'grade': np.random.choice([3, 4, 5], size=200),
        'year': np.random.choice([2010, 2011], size=200),
        'achv': np.random.uniform(-0.5, 0.5, 200),
        'achvz': np.random.uniform(0, 1, 200),
        'math': np.random.uniform(-0.5, 0.5, 200),
        'rla': np.random.uniform(-0.5, 0.5, 200),
        'totenrl': np.random.randint(200, 500, 200),
        'perasn': np.random.uniform(0, 0.01, 200),
        'perblk': np.random.uniform(0, 0.02, 200),
        'perhsp': np.random.uniform(0, 0.5, 200),
        'perind': np.random.uniform(0, 0.005, 200),
        'perwht': np.random.uniform(0, 1, 200),
        'perecd': np.random.uniform(0, 1, 200),
        'perell': np.random.uniform(0, 0.3, 200),
        'Locale3': np.random.choice(['Sub/Twn', 'Rural'], size=200),
        'DIST_FACTORS': np.random.uniform(0.05, 0.1, 200),
        'COUNTY_FACTORS': np.random.uniform(0.2, 0.3, 200),
        'HEALTH_FACTORS': np.random.uniform(-0.1, 0, 200)
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_df_sub(mock_df):
    return mock_df.drop(columns=['leaid', 'leanm', 'achv', 'Locale3', 'math', 'rla'])

def test_load_data(client, mock_df):
    with patch('school_performance_dashboard.get_df', return_value=mock_df), \
         patch('school_performance_dashboard.get_df_sub', return_value=mock_df):
        response = client.get('/api/load_data')
        assert response.status_code == 200
        assert json.loads(response.data) == "Data loaded successfully"

def test_filter_data(client, mock_df):
    with patch('school_performance_dashboard.get_df', return_value=mock_df):
        response = client.get('/api/filter_data?start=0&limit=2')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['data']) == 2
        assert data['pagination']['total_rows'] == 200

def test_get_features(client, mock_df_sub):
    with patch('school_performance_dashboard.get_df_sub', return_value=mock_df_sub):
        response = client.get('/api/get_features')
        assert response.status_code == 200
        features = json.loads(response.data)
        assert set(features) == set(mock_df_sub.columns)

def test_get_feature_ranges(client, mock_df_sub):
    with patch('school_performance_dashboard.get_df_sub', return_value=mock_df_sub):
        response = client.get('/api/get_feature_ranges')
        assert response.status_code == 200
        ranges = json.loads(response.data)
        assert set(ranges.keys()) == set(mock_df_sub.columns)
        for feature in ranges:
            assert set(ranges[feature].keys()) == set(['min', 'max', 'mean'])

def test_get_length(client, mock_df):
    with patch('school_performance_dashboard.get_df', return_value=mock_df):
        response = client.get('/api/get_length')
        assert response.status_code == 200
        length = json.loads(response.data)
        assert length == 200

@patch('models.lasso_cv')
def test_run_lasso(mock_lasso_cv, client, mock_df_sub):
    mock_metrics = pd.DataFrame({'Metrics': [0.1, 0.2, 0.8, 0.1, 0.01]}, 
                                index=['Mean Absolute Error', 'Mean Squared Error', 'R² Score', 'Best Alpha', 'Best Tolerance'])
    mock_coefficients = pd.DataFrame({'Coefficients': [0.5, 0.3, 0.2]}, index=mock_df_sub.columns[:3])
    mock_lasso_cv.return_value = (None, mock_metrics, mock_coefficients)

    with patch('school_performance_dashboard.get_df_sub', return_value=mock_df_sub):
        response = client.post('/api/run_lasso', json={'tolerance': 0.01, 'alpha': 0.1})
        assert response.status_code == 200
        result = json.loads(response.data)
        assert 'metrics' in result
        assert 'feature_importance' in result

@patch('models.lasso_cv')
def test_run_lasso_insufficient_data(mock_lasso_cv, client):
    # Mocking an empty DataFrame or DataFrame with insufficient rows
    insufficient_df = pd.DataFrame({'Columns': []})

    with patch('school_performance_dashboard.get_df_sub', return_value=insufficient_df):
        response = client.post('/api/run_lasso', json={'tolerance': 0.01, 'alpha': 0.1})
        assert response.status_code == 400
        result = json.loads(response.data)
        assert result['warning'] == 'Insufficient data for accurate predictions. Consider adding more data points.'
        assert result['data_points'] == 0

@patch('models.ext_trees')
def test_run_extra_trees(mock_ext_trees, client, mock_df_sub):
    mock_metrics = pd.DataFrame({'Metrics': [0.1, 0.2, 0.8]}, 
                                index=['Mean Absolute Error', 'Mean Squared Error', 'R² Score'])
    mock_model = MagicMock()
    mock_model.feature_importances_ = np.array([0.3, 0.3, 0.4])
    mock_ext_trees.return_value = (mock_model, mock_metrics)

    with patch('school_performance_dashboard.get_df_sub', return_value=mock_df_sub):
        response = client.post('/api/run_extra_trees', json={'n_estimators': 100})
        assert response.status_code == 200
        result = json.loads(response.data)
        assert 'metrics' in result
        assert 'feature_importances' in result

@patch('models.ext_trees')
def test_run_extra_trees_insufficient_data(mock_ext_trees, client):
    # Mocking an empty DataFrame or DataFrame with insufficient rows
    insufficient_df = pd.DataFrame({'Columns': []})

    with patch('school_performance_dashboard.get_df_sub', return_value=insufficient_df):
        response = client.post('/api/run_extra_trees', json={'n_estimators': 100})
        assert response.status_code == 400
        result = json.loads(response.data)
        assert result['warning'] == 'Insufficient data for accurate predictions. Consider adding more data points.'
        assert result['data_points'] == 0


@patch('models.lasso_cv')
def test_adjust_features(mock_lasso_cv, client, mock_df_sub):
    mock_metrics = pd.DataFrame({'Metrics': [0.1, 0.2, 0.8, 0.1, 0.01]}, 
                                index=['Mean Absolute Error', 'Mean Squared Error', 'R² Score', 'Best Alpha', 'Best Tolerance'])
    mock_coefficients = pd.DataFrame({'Coefficients': [0.5, 0.3, 0.2]}, index=mock_df_sub.columns[:3])
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.6])
    mock_lasso_cv.return_value = (mock_model, mock_metrics, mock_coefficients)

    with patch('school_performance_dashboard.get_df_sub', return_value=mock_df_sub):
        response = client.post('/api/adjust_features', json={
            'features': {col: mock_df_sub[col].mean() for col in mock_df_sub.columns},
            'target': 0.6
        })
        assert response.status_code == 200
        result = json.loads(response.data)
        assert 'adjusted_features' in result
        assert 'prediction' in result

def test_save_model(client):
    response = client.post('/api/save_model', json={
        'model_name': 'test_model',
        'model_state': {'param1': 1, 'param2': 2}
    })
    assert response.status_code == 200
    result = json.loads(response.data)
    assert result['message'] == "Model 'test_model' saved successfully."

def test_load_model(client):
    client.post('/api/save_model', json={
        'model_name': 'test_model',
        'model_state': {'param1': 1, 'param2': 2}
    })

    response = client.post('/api/load_model', json={'model_name': 'test_model'})
    assert response.status_code == 200
    result = json.loads(response.data)
    assert result['model_state'] == {'param1': 1, 'param2': 2}

def test_load_nonexistent_model(client):
    response = client.post('/api/load_model', json={'model_name': 'nonexistent_model'})
    assert response.status_code == 404
    result = json.loads(response.data)
    assert result['error'] == "Model not found."

@patch('school_performance_dashboard.log_queue')
def test_log_stream(mock_log_queue, client):
    mock_log_queue.get.return_value = "Test log message"
    
    response = client.get('/api/log_stream')
    assert response.status_code == 200
    assert b'data: Test log message' in next(response.response)