# Data Cleaning

Data cleaning, feature engineering, and pre-processing for the Digital Green Crop Yield Estimate Challenge. Please store data cleaning files for the model pipeline here.

This folder contains the following notebooks, to be executed in order within the model pipeline:

- **01.01-cleaning**: Dealing with missing values, outliers, parsing messy categorical columns, adding per_acre cols, basic feature engineering
- **01.02-clustering**: Adding cluster labels to the df (for k=2, k=3, k=4 and k=5 models, for now)
- **01.03-feature-engineering**: Creating new features to improve predictions (note: at this time, feature engineering is in 01.01)
- **01.04-scaling**: Encoding, scaling, and normalising
- **01.05-feature-selection**: Removing redundant or uninformative features
- **01.05b-outlier-classification**: Classification model for outliers vs non-outliers on yield-per-acre, with predictions for the test set.
- **01.06-dim-reduction**: Dimensionality reduction
- **01.07-Mean-Frequency-Encoding**: Encoding frequencies & Yield_per_Acre means for important features, + linear regression for important continuous variables. (NOTE: need to fix Num_CropbasalFerts & another bc values don't match between the train & test set, so currently the encodings are meaningless)
