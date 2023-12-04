-- fill in runtime
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

**The expected runtime is around XX minutes** (for a laptop with XXX config)



**Description of our submission**: 
1) **cleaning** (obtaining months from datetime columns, fixing suspected entry errors both for predictors and Yield, parsing messy categorical variables, imputation for missing values, and processing outliers by capping values) 
2) **feature engineering** (scaling values per Acre where relevant, second pass at capping outliers, adding binary variables marking missing values where meaningful, adding variables indicating the number of days between events, mode encoding & mean encoding, merging some months when low frequency, adding season variables, encoding how many fertilizers were reported to be used, and adding latitude, longitude & elevation for blocks)
3) **scaling** (one-hot encoding for categorical variables, min-max scaling for discrete variables, and standard scaling for continuous variables)
4) **initial feature selection** (dropping raw and very sparse variables)
5) **dimensionality reduction** (PCA with 21 components)
6) **clustering** (k-means for k ranging from 2 to 5, using different sets of features)
7) **2nd feature selection** (top_cols), selecting only features with significant predictive power (this is based on Shapley values, as examined from our model using a train-test split on the training set)
8) **modeling** using XGBoost, LightGBM and Cat… (?); training on the entire training dataset
9) **predictions** on the test set and exportation



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
