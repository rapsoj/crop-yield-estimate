import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

def reduce_dim(df):

    # Identify predictors with no missing data
    outcome_cols = ["Yield","Yield_per_Acre","New_Yield","New_Yield_per_Acre"]
    missing = ['SeedlingsPerPit','TransplantingIrrigationHours','TransIrriCost','StandingWater','1appDaysUrea',
               '2appDaysUrea','TransplantingIrrigationHours_per_Acre','TransIrriCost_per_Acre','TransIrriCost_per_Acre_capped',
               'Days_bw_Nurs_SowTransp','Days_bw_Nurs_Harv','Days_bw_Nurs_Till','NursingDate_ModeDiff',
               'Days_bw_Nurs_SowTransp_ModeDiff','Days_bw_Nurs_Till_ModeDiff','Days_bw_Nurs_Harv_ModeDiff','2appDaysUrea_MeanDiff',
               'TransplantingIrrigationHours_per_Acre_capped']
    X = df.drop(outcome_cols + missing, axis=1)

    # Initialize PCA with the number of components to retain
    n_components = 21  
    pca = PCA(n_components=n_components)

    # Fit and transform the data
    pca.fit(X)
    X_pca = pca.transform(X)

    # Create column names for the new principal components
    pc_columns = [f"PC{i+1}" for i in range(n_components)]

    # Create a DataFrame for the principal components
    df_pca = pd.DataFrame(data=X_pca, columns=pc_columns, index=X.index)

    # Concatenate the principal components DataFrame with your original DataFrame
    df_with_pca = pd.concat([df, df_pca], axis=1)

    return df_with_pca