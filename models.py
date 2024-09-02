
# Imports
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from IPython.display import display
import numpy as np

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize




# Lasso
    # 0.2/0.8 split (test=.2)
    # Gridsearch tunes param (alpha, tol), alpha may be hardset in future
    # Returns feature coefficient strengths and other Lasso performance metrics

def lasso_cv(df, tolerance, alpha):

    # Set target and data
    X = df.drop('achvz', axis=1)
    y = df['achvz']

    # , random_state = 64

    # train_test_split: test=.2, train=.8
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    # Setup Lasso
    lasso = Lasso()
    lasso.fit(X_train, y_train)

    # Lasso cross validation w/ tuning
    if (tolerance == "none" and alpha == "none"):
        param_grid = {
            'alpha' : [0.001, 0.01, 0.01, 0.1, 1],
            'tol' : [0.0001, 0.001, 0.01, 0.1, 1]
        }

    elif (tolerance != "none" and alpha == "none"):
        param_grid = {
            'alpha' : [0.00001, 0.0001, 0.001, 0.01],
            'tol' : [tolerance]
        }

    elif (tolerance == "none" and alpha != "none"):
        param_grid = {
            'alpha' :[alpha],
            'tol' : [0.0001, 0.001, 0.01, 0.1, 1]
        }

    elif (tolerance != "none" and alpha != "none"):
        param_grid = {
            'alpha' : [alpha],
            'tol' : [tolerance]
        }

    lasso_cv = GridSearchCV(lasso, param_grid, cv = 5, n_jobs = -1)
    lasso_cv.fit(X_train, y_train)
    y_pred2 = lasso_cv.predict(X_test)
    
    # Results
    lasso2 = lasso_cv.best_estimator_
    lasso2.fit(X_train, y_train)

    metric_names = ["Mean absolute Error", "Mean Squared Error", "R2 Score", "Alpha", "Tolerance"]
    values = [mean_absolute_error(y_test, y_pred2), mean_squared_error(y_test, y_pred2), r2_score(y_test, y_pred2), lasso_cv.best_params_['alpha'], lasso_cv.best_params_['tol']]
    df_t = pd.DataFrame(values, columns =['Metrics'], index=metric_names)

    feature_names = df.columns.tolist()
    feature_names.remove('achvz')

    df_t_coef = pd.DataFrame(lasso2.coef_, columns =['Coefficients'], index=feature_names)
    
        # Reintroduce in seperate sort method
    # df_t_coef_sorted = df_t_coef.iloc[np.argsort(np.abs(df_t_coef['Coefficients']))]

    return df_t, df_t_coef


def ext_trees(df):


    # Get target and data
    X = df.drop('achvz', axis=1)
    y = df['achvz']
    normalize(X)

    # 80/20 test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 64)

    # print(X)
    # print(y)
    # Scale data
    # scaler = RobustScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)

    # Fit model
    regressor = ExtraTreesRegressor(n_estimators=50)
    regressor.fit(X_train, y_train)

        
    # Tune params - Way Way Way too heavy on execution time
    
        # param_grid = {
        #     'n_estimators' : [250, 500, 1000],
        #     'max_depth' : [None, 10, 20, 30]
        # }

        # regressor_t = GridSearchCV(regressor, param_grid, cv = 5, n_jobs = -1)
        # regressor_t.fit(X_train, y_train)
        # print(regressor_t.best_estimator_)
        # regressor2 = regressor_t.best_estimator_
    

    # Cross-val
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=None)
    n_scores = cross_val_score(regressor, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')

        # Reintroduce later in separate function
    # print('ExtraTreesRegressor MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
    y_pred = regressor.predict(X_test)
    y_test = np.array(y_test)

    metric_names = ["Mean absolute Error", "Mean Squared Error", "R2 Score"]
    values = [mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)]
    df_t = pd.DataFrame(values, columns =['Metrics'], index=metric_names)

    print(df_t)

    # act_v_pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    return regressor, df_t