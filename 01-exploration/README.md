# Exploration

Exploring different hypotheses and issues with the data for the Digital Green Crop Yield Estimate Challenge. Please track the different model configurations and the corresponding RMSE here. Please track open questions and data cleaning ideas here.

### Open Questions

**Predictors**
- What are lag times between date of nursery establishment, sowing/transplanting, and threshing/harvesting?
- What are differences between months/seasons?
- Are results improved when dividing yield by area?
- Are there different crops? Can we detect them using unsupervised methods?
- Can we use past yields for that block as a predictor?
- Are there benefits to dimensionality reduction?
- Are there regional relationships between blocks?

**Models**
- Does XGBoost work as a model?
- Does XGBoost improve predictions with missing data?
- Can we change the XGBoost loss function to RMSE or RMSLE?
