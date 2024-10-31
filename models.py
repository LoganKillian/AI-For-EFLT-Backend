import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
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
            - Lasso: The trained Lasso model
            - pd.DataFrame: DataFrame containing performance metrics
            - pd.DataFrame: DataFrame containing the coefficients
            - float: Overall mean achievement score
    """
    logging.info("Starting Lasso cross-validation...")

    # Identifiers and target variables to exclude
    exclude_ids = ['leaid', 'leanm', 'year', 'grade', 'math', 'rla', 'achvz']

    # Get the tunable continuous features
    tunable_features = [
        col for col in df.columns 
        if (col not in exclude_ids)
    ]

    # Set target and data using only tunable features
    X = df[tunable_features]
    y = df['achvz']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the tunable features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Setup Lasso
    lasso = Lasso()

    # Default parameter grid for alpha and tolerance
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
    mse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    best_alpha = lasso_cv.best_params_['alpha']
    best_tolerance = lasso_cv.best_params_['tol']

    metrics = {
        "Mean Absolute Error": mae,
        "Mean Squared Error": mse,
        "R² Score": r2,
        "Best Alpha": best_alpha,
        "Best Tolerance": best_tolerance,
    }

    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Metrics'])

    # Coefficients
    coeffs_df = pd.DataFrame(lasso_cv.best_estimator_.coef_, columns=['Coefficients'], index=tunable_features)
    feature_importance = coeffs_df.reset_index().rename(columns={'index': 'feature'}).to_dict(orient='records')

    logging.info("Lasso cross-validation completed.")
    return lasso_cv.best_estimator_, metrics_df, feature_importance

def ext_trees(df, feature_adjustments=None, n_estimators=50): 
    """
    Perform ExtraTrees regression with proper handling of feature adjustments.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing features and target
        feature_adjustments (dict, optional): Dictionary of feature adjustments as percentages
        n_estimators (int, optional): Number of trees in the forest
    """
    logging.info("Starting ExtraTrees regression...")

    # Store original values
    original_df = df.copy()
    
    # Get tunable features
    exclude_ids = ['leaid', 'leanm', 'year', 'grade', 'math', 'rla', 'achvz']
    
    tunable_features = [
        col for col in df.columns 
        if (col not in exclude_ids)
    ]

    # Split data
    X = original_df[tunable_features]
    y = original_df['achvz']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit scaler on all training data
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Scale both training and test data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model on scaled original data
    regressor = ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        bootstrap=False,
        random_state=42
    )
    regressor.fit(X_train_scaled, y_train)
    
    # Get predictions for test data (for metrics)
    y_pred_test = regressor.predict(X_test_scaled)
    
    # Calculate feature importance similar to lasso_cv format
    feature_importance = [
        {'feature': feature, 'importance': importance}
        for feature, importance in zip(tunable_features, regressor.feature_importances_)
    ]
    feature_importance = sorted(feature_importance, key=lambda x: abs(x['importance']), reverse=True)
    
    # Handle adjustments if provided
    if feature_adjustments:
        # First scale the entire dataset
        X_full_scaled = scaler.transform(X)
        X_full_scaled_df = pd.DataFrame(X_full_scaled, columns=tunable_features)
        
        # Apply adjustments to the scaled values
        for feature, percentage in feature_adjustments.items():
            if feature in tunable_features:
                # Calculate the adjustment in scaled units
                adjustment = percentage / 100.0
                X_full_scaled_df[feature] *= (1 + adjustment)
                logging.info(f"Adjusted {feature} by {percentage}%")
        
        # Get predictions using adjusted scaled features
        y_pred_adjusted = regressor.predict(X_full_scaled_df)
        
        # Create comparison DataFrame with original scale values
        comparison_df = pd.DataFrame({
            'original_achvz': original_df['achvz'],
            'predicted_achvz': y_pred_adjusted
        })

        # Add original and adjusted feature values (in original scale)
        for col in tunable_features:
            comparison_df[f'original_{col}'] = original_df[col]
            # If the feature was adjusted, inverse transform the adjusted scaled values
            if col in feature_adjustments:
                # Create a temporary array with only this feature's adjusted values
                temp_scaled = np.zeros_like(X_full_scaled)
                temp_scaled[:, tunable_features.index(col)] = X_full_scaled_df[col]
                adjusted_values = scaler.inverse_transform(temp_scaled)[:, tunable_features.index(col)]
                comparison_df[f'adjusted_{col}'] = adjusted_values
            else:
                comparison_df[f'adjusted_{col}'] = original_df[col]
    else:
        # If no adjustments, use original scaled data for predictions
        X_full_scaled = scaler.transform(X)
        y_pred_adjusted = regressor.predict(X_full_scaled)
        
        comparison_df = pd.DataFrame({
            'original_achvz': original_df['achvz'],
            'predicted_achvz': y_pred_adjusted
        })
        
        # Add original values (no adjustments)
        for col in tunable_features:
            comparison_df[f'original_{col}'] = original_df[col]
            comparison_df[f'adjusted_{col}'] = original_df[col]

    # Calculate metrics using test data
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = root_mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    metric_names = ["Mean Absolute Error", "Mean Squared Error", "R² Score"]
    values = [mae, mse, r2]
    metrics_df = pd.DataFrame(values, columns=['Metrics'], index=metric_names)

    logging.info("ExtraTrees regression completed.")
    return regressor, metrics_df, comparison_df, feature_importance