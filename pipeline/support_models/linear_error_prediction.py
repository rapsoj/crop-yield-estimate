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
#os.chdir('/Users/jessicarapson/Documents/GitHub/crop-yield-estimate/')
#sys.path.insert(0, '/Users/jessicarapson/Documents/GitHub/crop-yield-estimate/pipeline')

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


#### CALCULATE LINEAR RELATIONSHIP BETWEEN ACRE AND YIELD ####

# Perform linear regression
slope, intercept = np.polyfit(df_train['Acre'], df_train['New_Yield'], 1)

# Calculate difference from line
predicted_y = slope * df_train['Acre'] + intercept
residuals = df_train['New_Yield'] - predicted_y
df_train['Residuals'] = residuals


#### TRAIN MODEL TO PREDICT LINEAR ERROR ####

# Split data
outcome_cols = ["Yield","Yield_per_Acre","New_Yield","New_Yield_per_Acre","Residuals"]
X, y = df_train.drop(outcome_cols, axis=1), df_train["Residuals"]

# Instantiate an XGBoost regressor model
best_params = {'alpha': 1, 'lambda': 0, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 150}
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, **best_params)

# Train the XGBoost model
xg_reg.fit(X, y)


#### RECALCULATE PREDICTION USING LINEAR MODEL AND PREDICTED LINEAR ERROR ####

# Make predictions
y_pred = xg_reg.predict(df[list(X.columns)])

# Calculate error when converting back into Per_Acre values
df['Linear_Yield_Prediction'] = slope * df['Acre'] + intercept + y_pred

# Export predictions
df[["Linear_Yield_Prediction"]].to_csv('pipeline/support_models/Linear_Yield_Prediction.csv', index=False)