import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def lasso_cv(df, tolerance=None, alpha=None):
    """
    Perform Lasso regression with cross-validation to find optimal parameters.

    Args:
        df (pd.DataFrame): Input DataFrame containing features and the target variable 'achvz'.
        tolerance (float, optional): Tolerance parameter for Lasso. It is the allowable error for convergence.
        alpha (float, optional): Alpha parameter for Lasso regularization. It is the regularization parameter.

    Returns:
        tuple: 
            - pd.DataFrame: DataFrame containing performance metrics (Mean Absolute Error, Mean Squared Error, R² Score, Best Alpha, Best Tolerance).
            - pd.DataFrame: DataFrame containing the coefficients of the features after Lasso regression.
    """
    logging.info("Starting Lasso cross-validation...")

    # Set target and data
    X = df.drop('achvz', axis=1)
    y = df['achvz']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Setup Lasso
    lasso = Lasso()

    # Default parameter grid
    param_grid = {
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1] if alpha is None else [alpha],
        'tol': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1] if tolerance is None else [tolerance]
    }
    
    # Lasso cross-validation with tuning
    logging.info(f"Using parameters: alpha={alpha}, tolerance={tolerance}")
    lasso_cv = GridSearchCV(lasso, param_grid, cv=5, n_jobs=-1)
    lasso_cv.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = lasso_cv.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    best_alpha = lasso_cv.best_params_['alpha']
    best_tolerance = lasso_cv.best_params_['tol']

    metrics = {
        "Mean Absolute Error": mae,
        "Mean Squared Error": mse,
        "R² Score": r2,
        "Best Alpha": best_alpha,
        "Best Tolerance": best_tolerance
    }

    # Log the metrics
    logging.info(
        f"Metrics after Lasso cross-validation:\n"
        f"Mean Absolute Error (MAE): {mae:.4f}\n"
        f"Mean Squared Error (MSE): {mse:.4f}\n"
        f"R² Score: {r2:.4f}\n"
        f"Best Alpha: {best_alpha}\n"
        f"Best Tolerance: {best_tolerance}"
    )

    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Metrics'])
    
    # Coefficients
    feature_names = df.columns.tolist()
    feature_names.remove('achvz')
    coeffs_df = pd.DataFrame(lasso_cv.best_estimator_.coef_, columns=['Coefficients'], index=feature_names)

    logging.info("Lasso cross-validation completed.")
    return lasso_cv.best_estimator_, metrics_df, coeffs_df

def ext_trees(df, n_estimators=50):
    """
    Perform ExtraTrees regression with hyperparameter tuning and cross-validation.

    Args:
        df (pd.DataFrame): Input DataFrame containing features and the target variable 'achvz'.
        n_estimators (int, optional): Number of trees in the forest. Default is 50.

    Returns:
        tuple:
            - ExtraTreesRegressor: Trained ExtraTrees model with the best parameters found by RandomizedSearchCV.
            - pd.DataFrame: DataFrame containing performance metrics (Mean Absolute Error, Mean Squared Error, R² Score).
    """
    logging.info("Starting ExtraTrees regression...")

    # Get target and data
    X = df.drop('achvz', axis=1)
    y = df['achvz']

    # Standardize data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 80/20 test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the base model
    regressor = ExtraTreesRegressor(n_estimators=n_estimators)

    logging.info("Tuning hyperparameters with RandomizedSearchCV...")

    # Define the hyperparameter space to search over
    param_distributions = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # RandomizedSearchCV for hyperparameter tuning
    random_search = RandomizedSearchCV(
        estimator=regressor,
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        scoring='negmean_absolute_error',
        random_state=42,
        n_jobs=-1 
    )

    # Fit RandomizedSearchCV to find the best parameters
    random_search.fit(X_train, y_train)

    # Get the best estimator from the randomized search
    best_regressor = random_search.best_estimator_
    logging.info(f"Best hyperparameters: {random_search.best_params_}")

    # Cross-validation for model evaluation
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
    n_scores = cross_val_score(best_regressor, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')

    # Predictions
    y_pred = best_regressor.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log performance metrics
    logging.info(
        f"ExtraTrees Regressor metrics:\n"
        f"MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}"
    )

    # Store metrics in DataFrame
    metric_names = ["Mean Absolute Error", "Mean Squared Error", "R² Score"]
    values = [mae, mse, r2]
    df_t = pd.DataFrame(values, columns=['Metrics'], index=metric_names)

    logging.info("ExtraTrees regression completed.")

    return best_regressor, df_t
