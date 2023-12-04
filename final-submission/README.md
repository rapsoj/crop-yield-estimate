-- feature explanation
-- review structure and filenames
-- try running on another pc to make sure it runs smoothly


**Hello!**

We are the Oxford Effective International Development Group. Thank you for your time in reviewing our model.



**This Read Me file covers**:
- submission folder structure
- how to run and expected runtime
- description of our process and training features



**Folder structure**:
- the “submission.ipynb” file is the only notebook that needs to be run. It includes both preprocessing and modeling.
- the “data” folder includes the datasets provided for this competition
- the “pipeline” folder includes the preprocessing .py files (cleaning.py, feature_engineering.py, scaling.py, feature_selection.py, dim_reduction.py and clustering.py)
- the “submission.csv” file is output by the model and includes IDs and Yield predictions for the test set.



**How to run**:
The only thing to do is running the submission.ipynb notebook, after replacing the path files in the first cell:
- os.chdir('/Users/yourname/Downloads/crop-yield-estimate-OxfordGroup/')
- sys.path.insert(0, '/Users/yourname/Downloads/crop-yield-estimate-OxfordGroup/pipeline')

**The expected runtime is around 3 minutes** (for a 2020 laptop with 8Gb RAM, Processor: AMD Ryzen 5 3500u, Graphic card: AMD Radeon vega 8 graphics)
- note: due to the nature of the models we used (XGBoost, LightGBM and CatBoost), the predictions will slightly vary with every run. Unfortunately, setting a seed does not allow for exact reproducibility, but the RMSE should be fairly stable. 



**Description of our submission:**: 
1) **cleaning** (obtaining months from datetime columns, fixing suspected entry errors both for predictors and Yield, parsing messy categorical variables, imputation for missing values, and processing outliers by capping values) 
2) **feature engineering** (scaling values per Acre where relevant, second pass at capping outliers, adding binary variables marking missing values where meaningful, adding variables indicating the number of days between events, mode encoding & mean encoding, merging some months when low frequency, adding season variables, encoding how many fertilizers were reported to be used, and adding latitude, longitude & elevation for blocks)
3) **scaling** (one-hot encoding for categorical variables, min-max scaling for discrete variables, and standard scaling for continuous variables)
4) **initial feature selection** (dropping raw and very sparse variables)
5) **dimensionality reduction** (PCA with 21 components)
6) **clustering** (k-means for k ranging from 2 to 5, using different sets of features)
7) **2nd feature selection**: the final list of features (top_cols) was selected using Recursive Feature Elimination with CV on our baseline XGBoost model.
9) **modeling** using XGBoost, LightGBM and CatBoost; training on the entire training dataset
    - note: we use per_Acre variables and our model predicts Yield_per_Acre. As post-processing, we revert back to raw Yield by multiplying the prediction by the Acre value
10) **predictions** on the test set and exportation

**More about our approach**:
We used an ensemble method that took the average of three predictions made using tree-based methods, specifically: XGBoost, CatBoost, and LightGBM. All three are tree-based methods, which are efficient for handling tabular data with non-linear relationships. We also used cross validation to train and test our model in order to reduce overfitting.
However, the most important part of our analysis lies at the data cleaning stage. We spent 95% of our time on data cleaning and feature engineering, which involved understanding the data and reading literature on which variables affect crop yield in Bihar, and India more generally. As a result, we learned about different agircultural traditions in North Bihar vs. South Bihar (which is more agriculturally productive and employs the Ahar Pyne agricultural system, leveraging channels and retention ponds to manage water resources and adapt to Bihar’s unpredictable weather). We also learned about the importance of the monsoon in Bihar agricultural cycles. Kharif crops, such as rice, are sown during the monsoon season from June to September and are watered by monsoon rainfall. These crops do well with high rain in Winter. We also learned about nitrogen cycles, fertilizer application methods, and irrigation techniques. This background research allowed us to realize that the region in which the crops was grown was important (North vs. South Bihar), as were the various dates on which key agricultural steps were taken and fertilisation choices. We engineered our data to reflect this, and selected the top variables using recursive feature elimination with cross validation.
Overall, we only spent around 5% of the time building the actual model (and only started fine-tuning the model two days before the deadline). The trick was to understand the data and avoid overfitting to the public test set.

**List of training features** (total number: 48):
- 'SeedlingsPerPit': was capped
- 'Ganaura': was capped
- 'CropOrgFYM':
- 'NoFertilizerAppln':
- 'BasalDAP':
- 'BasalUrea':
- '2appDaysUrea':
- 'Harv_hand_rent':
- 'Residue_length':
- 'TransplantingIrrigationHours_per_Acre':
- 'TransIrriCost_per_Acre':
- 'CropOrgFYM_per_Acre':
- 'BasalDAP_per_Acre':
- 'BasalUrea_per_Acre':
- '1tdUrea_per_Acre':
- 'Harv_hand_rent_per_Acre':
- 'TpIrrigationCost_Imputed_per_Acre':
- 'Days_bw_SowTransp_Harv':
- 'Days_bw_Harv_Thresh':
- 'NursingDate_ModeDiff':
- 'TillageDate_ModeDiff':
- 'HarvestDate_ModeDiff':
- 'ThreshingDate_ModeDiff':
- 'Num_LandPrepMethod':
- 'Num_CropbasalFerts':
- 'Num_TopDressFert':
- 'Latitude': added feature for each block
- 'Longitude': added feature for each block
- 'CropEstMethod_LineSowingAfterTillage':
- 'Threshing_method_machine':
- 'Stubble_use_plowed_in_soil':
- 'LandPrepMethod_FourWheelTracRotavator_True':
- 'LandPrepMethod_WetTillagePuddling_True':
- 'NursDetFactor_PreMonsoonShowers_True':
- 'NursDetFactor_LabourAvailability_True':
- 'FirstTopDressFert_DAP_True':
- 'HarvestMonth_November':
- 'ThreshingMonth_January':
- 'Block_Chehrakala':
- 'PCropSolidOrgFertAppMethod_Broadcasting':
- 'PCropSolidOrgFertAppMethod_SoilApplied':
- 'MineralFertAppMethod_1_Broadcasting':
- 'MineralFertAppMethod_1_SoilApplied': method of fertilizer application for the 1st application
- 'PC4': 4th component of the PCA
- 'PC10': 10th component of the PCA
- 'PC21': 21st component of the PCA
- 'top_shapley_k2_label_1': whether the data point is part of cluster 1 in the k=2 model (1) or not (0)
