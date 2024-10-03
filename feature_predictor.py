import numpy as np
import pandas as pd
from collections import namedtuple
import logging

"""Named tuple to define constraints of a feature: weight, min_val, max_val, and lock."""
FeatureConstraint = namedtuple('FeatureConstraint', ['weight', 'min_val', 'max_val', 'is_locked'])

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeaturePredictor:
    """Class used to predict feature adjustments for regression model."""

    def __init__(self, model, target, direction=1, current_pred=0, allowed_error=0.033,
                 early_exit=100, iteration_count=0):
        """Initializes the FeaturePredictor instance.

        Args:
            model: The regression model used for predictions.
            target: The target value we are trying to hit.
            direction: Indicates whether feature values should be increased
                or decreased to approach the target value (-1 or 1).
            current_pred: The current prediction of the model.
            allowed_error: The margin of error for determining if prediction is
                "close enough" to the target.
            early_exit: The threshold for exiting early if a certain number of epochs (iterations) is reached.
            iteration_count: Tracks the number of iterations performed.

        """
        self.model = model
        self.target = target
        self.direction = direction
        self.current_pred = current_pred
        self.allowed_error = allowed_error
        self.early_exit = early_exit
        self.iteration_count = iteration_count
        # Initialize empty map to hold features with their constraints
        self.feature_constraints_map = {}

        logging.info("FeaturePredictor initialized with target: %s", target)

    def is_within_allowed_error(self):
        """Checks if the current value is within the allowed error of the target value.

        Returns:
            bool: True if the current value is within the allowed error of the target, else False.

        """
        result = abs(self.current_pred - self.target) <= self.allowed_error
        logging.debug("Current prediction: %s, Target: %s, Within allowed error: %s", 
                      self.current_pred, self.target, result)
        return result
    
    def initialize_weights(self, weights, features_df, feature_constraints_map):
        """Initializes the weights of each feature.

        Args:
            weights: The regression weights for each feature.
            features_df: DataFrame holding the features.
            feature_constraints_map: A map holding weights, max_val, min_val values,
                and whether a feature is locked or not.

        """
        self.feature_constraints_map = {
            column: FeatureConstraint(
                weight=float(weights.iloc[i, 0]
                             * (features_df[column].max_val()
                                - features_df[column].min_val())
                                * self.reduction),
                min_val=features_df[column].min_val(),
                max_val=features_df[column].max_val(),
                is_locked=False
            )
            for i, column in enumerate(features_df.columns)
        }

        logging.info("Weights initialized for features: %s", {column: constraints.weight for column, constraints in self.feature_constraints_map.items()})

    def adjust_features(self, df):
        """Adjusts feature values based on their constraints.

        Args:
            df (pd.DataFrame): DataFrame containing the feature values to be adjusted.

        Returns:
            pd.DataFrame: The adjusted DataFrame with updated feature values.

        """
        for column in df:
            constraints = self.feature_constraints_map[column]

            # Continue if feature is not locked
            if constraints.is_locked == True:
                continue

            # Get value of min_val, max_val, and adjust current value by adding weight
            min_val = constraints.min_val
            max_val = constraints.max_val
            current_value = float(df.at[df.index[0], column])
            adjusted_value = current_value + constraints.weight

            # Replace current value if in range of min and max, else lock value at min or max
            if min_val < adjusted_value < max_val:
                df[column].replace(current_value, adjusted_value, inplace=True)
                logging.info("Feature %s adjusted from %s to %s", column, current_value, adjusted_value)
            elif adjusted_value <= min_val:
                df[column].replace(current_value, min_val, inplace=True)
                constraints.is_locked = True
                logging.info("Feature %s locked at minimum value %s", column, min_val)
            elif adjusted_value >= max_val:
                df[column].replace(current_value, max_val, inplace=True)
                constraints.is_locked = True
                logging.info("Feature %s locked at maximum value %s", column, max_val)

        logging.info("Current model prediction: %s", self.current_pred)
        return df

    def update_weights(self, damping_factor=0.5):
        """Updates weights based on the current prediction and target.

        Args:
            damping_factor (float): The factor by which to dampen the weight adjustments.

        """
        if self.current_pred < self.target:
            self.direction = 1
        else:
            self.direction = -1

        # Adjust the weights based on direction and damping_factor
        for feature, constraints in self.feature_constraints_map.items():
            weight, min_val, max_val, is_locked = constraints
            adjusted_weight = weight * self.direction * damping_factor
            self.feature_constraints_map[feature] = (adjusted_weight, min_val, max_val, is_locked)
            logging.info("Weights updated. Current prediction: %s, Direction: %s, Updated Weights: %s", 
              self.current_pred, self.direction, {feature: constraints.weight for feature, constraints in self.feature_constraints_map.items()})

    def set_locked_features(self, *features):
        """Sets specified features to be locked.

        Args:
            features (str): Variable length feature column names to be locked.

        """
        for feature in features:
            if feature in self.feature_constraints_map:
                constraints = self.feature_constraints_map[feature]
                self.feature_constraints_map[feature] = constraints._replace(is_locked=True)
                logging.info("Feature %s locked.", feature)
