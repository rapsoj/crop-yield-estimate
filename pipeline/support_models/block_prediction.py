#### PREPARE WORKSPACE ####

# Import system libraries
import os
import sys

# Import data cleaning libraries
import pandas as pd
import numpy as np

# Import machine learning libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Import preprocessing libraries
from preprocessing import cleaning
from preprocessing import feature_engineering
from preprocessing import scaling
from preprocessing import feature_selection
from preprocessing import dim_reduction
from preprocessing import clustering

# Import warning libraries
import warnings
warnings.filterwarnings("ignore")

# Set working directory
# os.chdir('/Users/jessicarapson/Documents/GitHub/crop-yield-estimate/')
# sys.path.insert(0, '/Users/jessicarapson/Documents/GitHub/crop-yield-estimate/pipeline')

# Preprocess data
train_path = "data/Train.csv"
test_path = "data/Test.csv"
df = cleaning.clean_data(train_path, test_path)
df = feature_engineering.get_features(df)
df = scaling.scale_features(df)
df = feature_selection.select_features(df)
df = dim_reduction.reduce_dim(df)
df = clustering.get_clusters(df)

# Split data into training and test sets
df_train = df[df['Yield'].isna() == False]
df_test = df[df['Yield'].isna() == True]


#### CREATE SEPARATE MODELS PER BLOCK ####

blocks = ["Block_Rajgir","Block_Jamui","Block_Mahua","Block_Khaira","Block_Noorsarai",
              "Block_Gurua","Block_Chehrakala","Block_Wazirganj","Block_Garoul"]

def run_xgboost(subset, best_params, df_train):
    outcome_cols = ["Yield","Yield_per_Acre","New_Yield","New_Yield_per_Acre"]
    X, y = df_train.drop(outcome_cols, axis=1), df_train["New_Yield_per_Acre"]
    # Instantiate an XGBoost regressor model
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3,
                              random_state=0, **best_params)
    # Train the XGBoost model on training data
    xg_reg.fit(X, y)
    # Predict for entire data
    y_pred = xg_reg.predict(df.drop(outcome_cols, axis=1))
    
    return y_pred


def run_xgboost_perblock():
    '''
        Runs XGBoost with optimal parameters for each Block, and returns the RMSE for each block (for the training set)
    '''
    # parameters obtained by running grid search on each block
    best_parameters = [{'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 125},#Rajgir
                       {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 75},#Jamui
                       {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50},#Mahua
                       {'learning_rate': 0.01, 'max_depth': 7, 'n_estimators': 125},#Khaira
                       {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 50},#Noorsarai
                       {'learning_rate': 0.01, 'max_depth': 7, 'n_estimators': 75},#Gurua
                       {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 10},#Chehrakala
                       {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 75},#Wazirgani
                       {'learning_rate': 0.001, 'max_depth': 5, 'n_estimators': 125}]#Garoul
    
    pred_df = pd.DataFrame(index=df.index)
    for block, best_params in zip(blocks, best_parameters):
        subset = df.loc[df[block]==True]
    
        # Split data into training and test sets
        df_train = subset[subset['Yield'].isna() == False]
        df_test = subset[subset['Yield'].isna() == True]
        
        # Append prediction to dataframe
        pred_df[block + "_Prediction"] = run_xgboost(subset, best_params, df_train)
        
    return pred_df

# Run model
per_block_pred = run_xgboost_perblock()

# Select prediction based on data block
block_pred = pd.concat([per_block_pred, df[blocks]], axis=1)
for block in blocks:
    block_pred.loc[block_pred[block], ['Block_Prediction']] = block_pred[block_pred[block]][block + '_Prediction']

# Export predictions
block_pred['Block_Prediction'].to_csv('pipeline/support_models/Block_Prediction.csv', index=False)