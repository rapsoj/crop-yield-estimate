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

#### TRAIN ORIGINAL MODEL ####

# Split data
outcome_cols = ["Yield","Yield_per_Acre","New_Yield","New_Yield_per_Acre"]
X, y = df_train.drop(outcome_cols, axis=1), df_train["New_Yield_per_Acre"]

# Instantiate an XGBoost regressor model
best_params = {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 100}
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, **best_params)

# Train the XGBoost model
xg_reg.fit(X, y)


#### CALCULATE ERRORS ####

# Make predictions
y_pred = xg_reg.predict(X)

# Create new dataframe to store errors
data = X
data['Error'] = y - y_pred
data['Error_Scaled'] = (data['Error'] - data['Error'].mean()) /  data['Error'].std()


#### TRAIN ERROR PREDICTION MODEL ####

# Splitting the data into train and test sets
outcome_cols = ["Yield","Yield_per_Acre","New_Yield","New_Yield_per_Acre"]
X_error, y_error = df_train.drop(outcome_cols, axis=1), data["Error_Scaled"]

# Defining the XGBoost regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42,
                            colsample_bytree=0.3, **best_params)

# Training the XGBoost regressor
xgb_model.fit(X_error, y_error)

# Making predictions on the test set
y_pred_final = xgb_model.predict(df[list(X_error.columns)])
df["Error_Prediction"] = y_pred_final

# Export predictions
df[["Error_Prediction"]].to_csv('pipeline/support_models/Error_Prediction.csv', index=False)