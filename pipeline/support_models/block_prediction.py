def run_xgboost(subset, best_params, df_train):
    X, y = df_train.drop(["Yield","Yield_per_Acre","New_Yield","New_Yield_per_Acre"], axis=1), df_train["New_Yield_per_Acre"]
    # Instantiate an XGBoost regressor model
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, random_state=0,
                         **best_params)
    # Train the XGBoost model
    xg_reg.fit(X, y)
    # Predict on the train set
    y_pred = xg_reg.predict(X)
    # RMSE (Root Mean Squared Error)
    indices = list(y.index)
    rmse = mean_squared_error(df_train.loc[indices]["New_Yield"],
                          y_pred * df_train.loc[indices]["Acre"], squared=False)
    print("Root Mean Squared Error on train set:", rmse.round(3))
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
    blocks = ["Block_Rajgir","Block_Jamui","Block_Mahua","Block_Khaira","Block_Noorsarai","Block_Gurua",
              "Block_Chehrakala","Block_Wazirganj","Block_Garoul"]
    for block, best_params in zip(blocks, best_parameters):
        print(block)
        subset = df.loc[df[block]==True]
        # Split data into training and test sets
        df_train = subset[subset['Yield'].isna() == False]
        df_test = subset[subset['Yield'].isna() == True]
        run_xgboost(subset, best_params, df_train)
        print('\n')
run_xgboost_perblock()