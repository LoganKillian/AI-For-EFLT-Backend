from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler, normalize
import tqdm

from Predictor import FeaturePredictor
import models


app = Flask(__name__)
CORS(app)



@app.route('/api/init_data', methods=['GET'])
def init_data():
    global df 
    global df_sub

    df = pd.read_csv('AL_Dist.csv')
    df_sub = pd.read_csv('AL_Dist.csv')

    drop_cols = [
        'leaid', 
        'achv', 
        'Locale3', 
        'math', 
        'rla'
    ]

    df_sub.drop(columns=drop_cols, inplace=True)

    
    for column in df_sub.columns:
        if df_sub[column].astype(str).str.contains(r'[0-9.-]', regex=True).any():
            pass
        else:
            df_sub.drop(columns=column, inplace=True)



    for column in df_sub.columns:
        if df_sub[column].isnull().any():
            mean = df_sub[column].mean()
            # df_sub[column].fillna(mean, inplace=True)
            df_sub.fillna({column: mean}, inplace=True)
 
    for column in df_sub.columns:
        if df_sub[column].dtype == 'object':
            df_sub[column] = pd.to_numeric(df_sub[column], errors='coerce')

    s = "okay"
    return jsonify(s)



@app.route('/api/get_data', methods=['GET'])
def get_data():

    start = int(request.args.get('start'))
    rows = int(request.args.get('limit'))

    data_slice = df.iloc[start:start + rows]
    data_slice.insert(0, 'Index', (data_slice.index + 1))
    print(data_slice)

    placeholder = ""
    data_slice = data_slice.where(pd.notna(data_slice), placeholder )
    data_json = data_slice.to_dict(orient='records')


    
    return jsonify(data_json)


@app.route('/api/get_length', methods=['GET'])
def get_length():
    return jsonify(len(df))


@app.route('/api/lasso', methods=['POST'])
def lasso():
    
    global df_sub_coef
    global df_sub
    global df_red_sub
    global df_red_coef

    data = request.json
    alpha = data.get('alpha')
    tolerance = data.get('tolerance')
    reduction = float(data.get('reduction'))

    if (alpha == ''):
        alpha="none"
    else:
        alpha = float(alpha)

    if (tolerance == ''):
        tolerance="none"
    else:
        tolerance = float(tolerance)


    df_sub_t, df_sub_coef = models.lasso_cv(df_sub, tolerance, alpha)

    df_red_coef = df_sub_coef[abs(df_sub_coef['Coefficients']) > reduction]
    df_red_sub = df_sub[df_sub.columns[df_sub.columns.isin(df_red_coef.index)]]
    df_red_sub['achvz'] = df_sub['achvz']

    print(df_sub_coef)
    print(df_sub_t)
    print(df_red_sub)
    
    response = {
        'metrics' : df_sub_t.to_dict(orient='index'),
        'coefficients' : df_red_coef.to_dict(orient='index')
    }



    return jsonify(response)


@app.route("/api/run_predictor", methods=['POST'])
def run_predictor():

    global pred
    global mod_x_row
    global pred_results
    global x
    global first_df
    global df_red_sub
    global df_red_coef

    print("in\n\n")

    response = request.json

    print("out\n\n")

    school = int(response.get('school')) - 1
    ee = float(response.get('earlyExit'))
    allErr = float(response.get('allowedError'))
    target = float(response.get('targetVal'))
    lock_feat = response.get('lock')
    


    # if (reg.isin(new_models)): # IMPLEMENT LAZYPREDICT
    #     mod = reg
    # else:
    #     mod = models.ext_trees(df)
    
    mod, metrics = models.ext_trees(df_red_sub)
    pred = FeaturePredictor(regressor=mod, target=target)

    # Initializes starting feature weights / value to be changed
    x = df_red_sub.drop('achvz', axis=1)
    pred.init_weights(df_red_coef, x, lock_feat)
    pred.allowed_error = allErr
    pred.early_exit = ee
    pred_row = df_red_sub.loc[[school]]    
    pred.curr_val = pred_row.loc[pred_row.index[0]]['achvz']
    
    # Original row
    x_row = pred_row.drop('achvz', axis=1)

    # Modified row
    mod_x_row = x_row.copy()

    # Storage
    pred_results = x_row.copy()
    pred_results = pred_results.rename(index={0: '0'})

    scaler = StandardScaler()
    scaler.fit(x)
    first_df = pd.DataFrame(scaler.transform(mod_x_row), columns=x.columns)

    print(mod_x_row)
    print(x)

    data = {
        'pred': pred.curr_val,
        'index': pred.curr_count,
        'df': first_df.to_dict(orient='list'),
        'metrics': metrics.to_dict(orient='index')
    }


    return jsonify(data)


@app.route("/api/fetch_pred", methods=['GET'])
def fetch_pred():

    global pred
    global mod_x_row
    global pred_results

    if((pred.match() == False) and (pred.curr_count < pred.early_exit)):

        # Set pol and stretch features
        mod_x_row = pred.stretch_feat(mod_x_row)

        pred_results = pred_results._append(mod_x_row, ignore_index=True)
        pred_results = pred_results.rename(index={pred.curr_count: pred.curr_count})

        mean_predictions = pred.regressor.predict(mod_x_row)

        # Set new achvz progress
        pred.curr_val = float(mean_predictions)
        pred.set_pol()
        
        # Iterate early exit count
        pred.curr_count += 1        

        scaler = StandardScaler()
        scaler.fit(x)
        new_df = pd.DataFrame(scaler.transform(mod_x_row), columns=x.columns)
        first_df.loc[1] = new_df.iloc[0]
        final_df = first_df

        data = {
            'pred': pred.curr_val,
            'index': pred.curr_count,
            'df': final_df.to_dict(orient='list')
        } 
     
        return jsonify(data)
    
    else: 

        data = {
            'pred': 111,
            'index': 111,
            'df': 111
        } 
        print(pred_results)
        
        return jsonify(data)
    





if __name__ == '__main__':
    app.run(debug=True)



