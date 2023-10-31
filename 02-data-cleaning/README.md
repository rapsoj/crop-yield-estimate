# Data Cleaning

Data cleaning, feature engineering, and pre-processing for the Digital Green Crop Yield Estimate Challenge. Please store data cleaning files for the model pipeline here.

This folder contains the following notebooks, to be executed in order within the model pipeline:

- **01.01-cleaning**: Dealing with missing values, outliers, parsing messy categorical columns, adding per_acre cols
- **01.02-clustering**: Adding cluster labels to the df (for k=2, k=3, k=4 and k=5 models, for now)
- **01.03-feature-engineering**: Creating new features to improve predictions
- **01.04-scaling**: Encoding, scaling, and normalising
- **01.05-feature-selection**: Removing redundant or uninformative features
- **01.06-dim-reduction**: Dimensionality reduction



Note on cleaned_fulldf_withclusters.csv (sorry for the horrible name): 
- For modeling on raw variables, drop: "TransIrriCost_per_Acre","Ganaura_per_Acre","CropOrgFYM_per_Acre","BasalDAP_per_Acre","BasalUrea_per_Acre","1tdUrea_per_Acre","2tdUrea_per_Acre","Harv_hand_rent_per_Acre","Yield_per_Acre"
- For modeling on per_acre, drop: "TransIrriCost","Ganaura","CropOrgFYM","BasalDAP","BasalUrea","1tdUrea","2tdUrea","Harv_hand_rent","Yield"
