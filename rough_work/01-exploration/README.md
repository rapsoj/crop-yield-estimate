# Exploration

Exploring different hypotheses and issues with the data for the Digital Green Crop Yield Estimate Challenge. Please track open questions, literature reviews, key argicultural terminology, and potential solutions to the challenge here.

The following Google Sheets documents are available for tracking data exploration progress:
- **Literature Review**: Academic and technical literature on current best practices and proposed methodology for crop yield forecasting
- **Agricultural Terminology**: Key terms and definitions for understanding crop yield forecasting
- **Variable Descriptions**: Summaries of all variables in the Crop Yield Estimate challenge

[Literature Review and Agricultural Terminology](https://docs.google.com/spreadsheets/d/1rkmqYmPFRBwvi-g2wKKVJ9ZElrss3gw5C6AgDLTZF8k/edit?usp=sharing)

[Variable Descriptions](https://docs.google.com/spreadsheets/d/1ASWzNjuvdYaqWgYlU4L_7gABIkig0wKU8GyFhQ19rx4/edit#gid=174692797)

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
