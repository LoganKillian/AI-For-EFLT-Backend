


import numpy as np
import pandas as pd

# Predictor object for pred_features
class FeaturePredictor:

    reduction = .75

    def __init__(self,
                 regressor,
                 target,
                 polarity = 1,
                 curr_val = 0,
                 lock = pd.DataFrame(),
                 allowed_error=.033, 
                 early_exit=100,
                 curr_count=0
                 ):

        self.curr_val = curr_val
        self.polarity = polarity
        self.regressor = regressor
        self.allowed_error = allowed_error
        self.early_exit = early_exit
        self.target = target
        self.lock = lock
        self.curr_count = curr_count

    # Evaluates to TRUE if value is within Predictor margin of error
    def match(self):

        if (self.target - self.allowed_error <= self.curr_val <= 
            self.target + self.allowed_error):
            
            return True
        else:
            return False
        
    # Initializes weights
    def init_weights(self, coef, df, lock_feat):

        ranges = df.apply(lambda x: x.max() - x.min())
        self.weights = np.zeros(df.shape[0])

        self.lock = pd.DataFrame(columns= df.columns, index= ["Weight", "Min", "Max", "Lock"])
        self.lock.loc["Lock"] = 1
        
        for i, row in enumerate(coef.index):

            min = df[df.columns[i]].min()
            max = df[df.columns[i]].max()
            range = max-min

            self.lock.iloc[0][i] = float((coef.iloc[i][coef.columns[0]])*(range)*self.reduction)
            self.lock.iloc[1][i] = min
            self.lock.iloc[2][i] = max

        for i, var in enumerate(lock_feat):

            self.lock.loc["Lock"][var] = 0
            

    # Takes in single rowed dataframe, modulates feature values
    def stretch_feat(self, df):

        for i, column in enumerate(df.columns):

            if (self.lock.loc["Lock"][column]) == 0:
                continue
            else:
                sum = float(df.iloc[0][i]) + self.lock.iloc[0][i]
                min = self.lock.iloc[1][i]
                max = self.lock.iloc[2][i]

                if (sum > min and sum < max):
                    df[column] = df[column].replace(df.loc[df.index[0]][column], sum)
                elif (sum < min):
                    df[column] = df[column].replace(df.loc[df.index[0]][column], min)
                    self.lock.loc["Lock"][column] = 0
                elif (sum > max):
                    df[column] = df[column].replace(df.loc[df.index[0]][column], max)
                    self.lock.loc["Lock"][column] = 0
                

        return df

    # Flips weight values if overshoot target
    def set_pol(self):
        if (self.curr_val < self.target and self.polarity == -1 or 
            self.curr_val > self.target and self.polarity == 1):

            self.polarity *= -1
            self.modify_weights()



    # Flips weights
    def modify_weights(self):
        for i, weight in enumerate(self.lock.loc["Weight"]):
            self.lock.iloc[0][i] = weight * (-1)  * (.5)



    def init_lock(self, df):
        pass

    # Check valid ranges, change flags 1->0 if outside range
    def modify_lock(self, inp):
        pass

    
    





    # # Flags whether the current value needs to increase or decrease, 1 -> target is higher, -1 -> target is lower        
    # def set_pol(self):

    #     if (self.target > self.curr_val):
    #         self.polarity = 1
    #     else:
    #         self.polarity = -1


    # def pred():
    #     pass