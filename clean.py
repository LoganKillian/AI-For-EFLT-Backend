# Imports
import numpy as np
import pandas as pd
from IPython.display import display
import re
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


############################################################################### Dataframe Init ###############################################################################

# Class object for categorical features
class Feature_c:
    def __init__(self, name, data):
        self.name = name
        self.data = data

# Class object for numerical features
class Feature_n:
    def __init__(self, name, data):
        self.name = name
        self.data = 0


# Categorical variables
cat_var = ['grade', 'year', 'leanm', 'Locale4', 'BlackBeltSm', 'FoodDesert', 'CT_EconType']
cat_obj = [None] * len(cat_var)

# Nominal variables
cat_str = ['leanm','Locale4','CT_EconType']

# Numerical variables, cannot select for more than 100% in subset
num_var = ['perasn','perblk','perwht','perind','perhsp', 'perecd', 'perell']
num_obj = [None] * len(num_var)


# Initializes changeable features for subset creation
    # Refactor - make single loop, reduce redundancy
def init_df(df):
    temp_df = df.copy()
    # temp_df.set_index('leaid')

    # Always drop
        # leaid -> noise, achv -> alternate of predicted metric, Locale3 -> alternate of Locale4, math -> 1 to 1 of achvz, rla -> 1 to 1 of achvz
    drop_cols = [
        'leaid', 
        'achv', 
        'Locale3', 
        'math', 
        'rla'
    ]

    temp_df.drop(columns=drop_cols, inplace=True)



    # Drop columns if they are fully empty. Should be: LOCALE_VARS, DIST_FACTORS, HEALTH_FACTORS, COUNTY_FACTORS
    for column in temp_df.columns:
        if ~temp_df[column].notna().any():
            temp_df.drop(columns=column, inplace=True)

    print(temp_df)



    for i, var in enumerate(cat_var):
        name = cat_var[i]

        data = np.array(temp_df[name].unique())
        data = remove_nan(data)

        cat_obj[i] = Feature_c(name, data)
        
    for j, var in enumerate(num_var):
        name = num_var[j]
        
        num_obj[j] = Feature_n(name, data)

    
    features = [cat_obj, num_obj]
    
    for k in cat_str:
        temp_df[k] = temp_df[k].str.lower()

    
    display(temp_df)

    return temp_df, features


# Formats input - Removes whitespace, ignores case, converts type
def format(str):
    str = str.split(',')
    formatted_array = []
    
    for i in str:
        if re.search(r'\d', i):
            formatted_element = int(i.strip())
        else:    
            formatted_element = i.lower().strip()
        formatted_array.append(formatted_element)

    return formatted_array


def format2(arr):

    print(arr)

    for i, val in enumerate(arr):
            
        print(val)

        if re.search(r'^[^A-Za-z]*$', val):
            arr[i] = int(val)
        else:
            arr[i] = val.lower()
    return arr 


# Removes 'nan' values in feature data
def remove_nan(arr):
     return np.array([item for item in arr if str(item).lower() != 'nan'])




############################################################################### Cleaning ############################################################################### 

# Two methods per issue
    # Nominal values - Either drop them or encode them
    # Empty values - Either drop them or populate them

# Binary encoding for categorical variables
def bin_encode(df):
    pass



# Drop nominal values - Extremely sloppy, refactor in the future
def drop_nom(df):
    for column in df.columns:
        if df[column].astype(str).str.contains(r'[0-9.-]', regex=True).any():
            pass
        else:
            df.drop(columns=column, inplace=True)


# Fill nan values
def mean_sub(df):
    for column in df.columns:

        if df[column].isnull().any():
             
            mean = df[column].mean()
            df[column].fillna(mean, inplace=True)


# Drop nan values in groups, hinge on specified/default parameter
def drop_gap(df):
    pass


# Print current subset feature object data
def prt_feat_data(features):
    for i, group in enumerate(features):
        for j, feat in enumerate(group):
            name = feat.name
            data = feat.data
            print(f"Feature Name: {name}\tFeature Data: {data}\n")

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

############################################################################### Locale4 Encoding ###############################################################################

# One-Hot Encoding for Locale4
def encode_locale4(df):
    # Fill missing values with 'Unknown'
    df['Locale4'] = df['Locale4'].fillna('Unknown')  
    df['Locale4'] = df['Locale4'].str.lower()  

    # One-Hot Encoding for Locale4
    encoder = OneHotEncoder(sparse_output=False)  
    encoded_cols = encoder.fit_transform(df[['Locale4']])
    
    # Create a new DataFrame with the encoded columns
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(['Locale4']))
 

    df_encoded = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1).drop(columns=['Locale4'])

    return df_encoded
############################################################################### CT_EconType Encoding ###############################################################################
def encode_ct_econtype(df):
    # Fill missing values with 'Unknown'
    df['CT_EconType'] = df['CT_EconType'].fillna('Unknown')
    df['CT_EconType'] = df['CT_EconType'].str.lower()  

    # One-Hot Encoding for CT_EconType
    encoder = OneHotEncoder(sparse_output=False)
    encoded_cols = encoder.fit_transform(df[['CT_EconType']])

    # Create a new DataFrame with the encoded columns
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(['CT_EconType']))

    
    df_encoded = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1).drop(columns=['CT_EconType'])

    return df_encoded
############################################################################### leanm Encoding ###############################################################################
def encode_leamn_with_frequency(df):
    # frequency encoding
    freq_encoding = df['leanm'].value_counts() / len(df)
    

    df['leanm_encoded'] = df['leanm'].map(freq_encoding)
    
    return df

############################################################################### Grade Encoding ###############################################################################
def encode_grade(df):
    # Fill missing values and ensure data is in string format (if needed)
    df['grade'] = df['grade'].fillna('Unknown').astype(str)

    # Define ordinal categories in the desired order (3 to 8)
    grade_categories = ['3', '4', '5', '6', '7', '8']

    # Apply Ordinal Encoding
    encoder = OrdinalEncoder(categories=[grade_categories])
    df['grade_encoded'] = encoder.fit_transform(df[['grade']])


    return df
############################################################################### Year Encoding ###############################################################################
def encode_year(df):
    # Fill missing values with 'Unknown' or appropriate value (if necessary)
    df['year'] = df['year'].fillna('Unknown')
    
    # Create an OrdinalEncoder instance
    encoder = OrdinalEncoder()
    df['year_encoded'] = encoder.fit_transform(df[['year']])
    
    return df

# Misc Notes


# priorities when coming back, refactor pre-processing method - fill missing values: ALL? or minor?
# remove big chunks before subset selection - static/init "drop_gaps()", "df_init()"
# then mean substitution for remaining holes - after/dependent "mean_sub()"
# Preproc() -> df_init(), ____Subset____, mean_sub() (and encoding), Models, graphs
# encode
# ct_econtype
# Locale4
# leanm
#
# New prios: 
# encode nominals
# drop gappy?
# Integrate graphing / refactor graphing
#
#
#
#
#
# When only using 1 value of nominal value, can drop column, otherwise binary encode 