import pytest
import json
import pandas as pd
from unittest.mock import patch, MagicMock
from main import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

"""Test for the /api/init_data route."""
def test_init_data(client):
    with patch('main.pd.read_csv') as mock_read_csv:
        mock_df = pd.DataFrame({
            'leaid': [1, 2],
            'achv': [3, 4],
            'Locale3': [5, 6],
            'math': [7, 8],
            'rla': [9, 10],
            'OtherCol': [11, 12]
        })
        mock_read_csv.return_value = mock_df

        response = client.get('/api/init_data')
        assert response.status_code == 200
        assert response.json == "okay"

"""Test for the /api/get_data route."""
def test_get_data(client):
    with patch('main.df', pd.DataFrame({
        'Column1': ['a', 'b', 'c'],
        'Column2': [1, 2, 3]
    })):
        response = client.get('/api/get_data', query_string={'start': 0, 'limit': 2})
        assert response.status_code == 200
        data = response.json
        assert len(data) == 2
        assert data[0]['Column1'] == 'a'

"""Test for the /api/get_data_by_district route."""
def test_get_data_by_district(client):
    with patch('main.df', pd.DataFrame({
        'Column1': ['a', 'b', 'c', 'd'],
        'Column2': [1, 2, 3, 4],
        'leanm': ['District1', 'District2', 'District1', 'District3'],
        'leaid': [101, 102, 101, 103]
    })):
        # Test without filtering
        response = client.get('/api/get_data', query_string={'start': 0, 'limit': 3})
        assert response.status_code == 200
        data = response.json
        assert len(data) == 3
        assert data[0]['Column1'] == 'a'

        # Test with district name filtering
        response = client.get('/api/get_data', query_string={'start': 0, 'limit': 3, 'district_name': 'District1'})
        assert response.status_code == 200
        data = response.json
        assert len(data) == 2
        assert data[0]['Column1'] == 'a'
        assert data[1]['Column1'] == 'c'

        # Test with district ID filtering
        response = client.get('/api/get_data', query_string={'start': 0, 'limit': 3, 'district_id': '101'})
        assert response.status_code == 200
        data = response.json
        assert len(data) == 2
        assert data[0]['Column1'] == 'a'
        assert data[1]['Column1'] == 'c'

"""Test for the /api/get_length route."""
def test_get_length(client):
    with patch('main.df', pd.DataFrame({
        'Column1': [1, 2, 3, 4, 5]
    })):
        response = client.get('/api/get_length')
        assert response.status_code == 200
        assert response.json == 5

"""Test for the /api/lasso route."""
def test_lasso(client):
    mock_data = {
        'alpha': '0.01',
        'tolerance': '0.001',
        'reduction': '0.1'
    }

    mock_lasso_df = pd.DataFrame({
        'Coefficients': [0.5, 0.2, 0.05],
        'Column': ['feat1', 'feat2', 'feat3']
    })

    with patch('main.models.lasso_cv', return_value=(mock_lasso_df, mock_lasso_df)):
        response = client.post('/api/lasso', data=json.dumps(mock_data),
                               content_type='application/json')
        assert response.status_code == 200
        response_json = response.json
        assert 'metrics' in response_json
        assert 'coefficients' in response_json

"""Test for the /api/run_predictor route."""
def test_run_predictor(client):
    mock_data = {
        'school': 1,
        'earlyExit': 0.1,
        'allowedError': 0.01,
        'targetVal': 0.5,
        'lock': []
    }

    with patch('main.models.ext_trees', return_value=(MagicMock(), MagicMock())):
        response = client.post('/api/run_predictor', data=json.dumps(mock_data),
                               content_type='application/json')
        assert response.status_code == 200
        response_json = response.json
        assert 'pred' in response_json
        assert 'metrics' in response_json

"""Test for the /api/fetch_pred route."""
def test_fetch_pred(client):
    with patch('main.pred.match', return_value=False), \
         patch('main.pred.stretch_feat', return_value=MagicMock()), \
         patch('main.pred.regressor.predict', return_value=[0.75]):

        response = client.get('/api/fetch_pred')
        assert response.status_code == 200
        response_json = response.json
        assert 'pred' in response_json
        assert response_json['pred'] == 0.75
