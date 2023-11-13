#!/usr/bin/env python
# coding: utf-8

# #### Import

# In[134]:


#!pip install pingouin


# In[135]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import calendar
#import pingouin as pg
from datetime import datetime
pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings("ignore")


# In[136]:


from google.colab import drive
drive.mount('/content/gdrive')

# directory
get_ipython().run_line_magic('cd', "'/content/gdrive/My Drive/Oxford/ML_for_Social_Good'")

# import
train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")

train["Set"] = "train"
test["Set"] = "test"

df = pd.concat([train, test])
print(df.shape)


# #### Basic adjustments

# In[137]:


datetime_cols = ["CropTillageDate","RcNursEstDate","SeedingSowingTransplanting","Harv_date","Threshing_date"]

for col in datetime_cols:
  df[col] = pd.to_datetime(df[col])#.dt.date


# In[138]:


# One row has Jamui as district but Gurua as Block, which is a Gaya block -- correcting its District
df.loc[(df["District"]=="Jamui") & (df["Block"]=="Gurua")].index
df.loc[2177,"District"] = "Gaya"


# #### Outliers

# In[139]:


# OUTLIERS -- first pass

# SeedlingsPerPit has two extreme outliers (800 seedlings & 442 seedlings) -- replacing with the next max value (22)
#df["SeedlingsPerPit"] = df["SeedlingsPerPit"].replace(800,22).replace(442,22)
df.loc[df["SeedlingsPerPit"]>22, "SeedlingsPerPit"] = 22

# TransplantingIrrigationHours -- capping at 450
df.loc[df["TransplantingIrrigationHours"]>450, "TransplantingIrrigationHours"] = 450

# TransIrriCost have several extreme outliers (e.g. 6000 rupees for an average sized land) -- capping at 3000
df.loc[df["TransIrriCost"]>3000, "TransIrriCost"] = 3000

# Ganaura -- making a capped version at 50 (already way above the upper fence), and leaving the raw variable to compare
df.loc[df["Ganaura"]>50, "Ganaura_capped"] = 50

# 1appDaysUrea -- replacing extreme outlier with the next max value
df["1appDaysUrea"] = df["1appDaysUrea"].replace(332,75)

# Harv_hand_rent -- capping at 20000 (there are 2 values above that) (upper fence is at 1500, so 20000 is conservative)
df.loc[df["Harv_hand_rent"]>20000, "Harv_hand_rent"] = 20000


# #### Per_Acre columns

# In[140]:


# PER-ACRE COLUMNS

list_cols = ["TransIrriCost","Ganaura","CropOrgFYM","BasalDAP","BasalUrea","1tdUrea","2tdUrea","Harv_hand_rent","Yield"]

for col in list_cols:
  label = str(col) + "_per_Acre"
  df[label] = df[col] / df["Acre"]


# In[141]:


# Re-ordering columns and dropping the non-standardized ones
df = df[['ID', 'Set', 'District', 'Block', 'LandPreparationMethod', 'CropTillageDate', 'CropTillageDepth','CropEstMethod', 'RcNursEstDate',
         'SeedingSowingTransplanting','SeedlingsPerPit', 'NursDetFactor', 'TransDetFactor','TransplantingIrrigationHours', 'TransplantingIrrigationSource',
         'TransplantingIrrigationPowerSource', "TransIrriCost", 'TransIrriCost_per_Acre', 'StandingWater','OrgFertilizers', 'Ganaura', 'Ganaura_per_Acre', 'CropOrgFYM',
         'CropOrgFYM_per_Acre', 'PCropSolidOrgFertAppMethod', 'NoFertilizerAppln', 'CropbasalFerts', "BasalDAP", 'BasalDAP_per_Acre', "BasalUrea", 'BasalUrea_per_Acre',
         'MineralFertAppMethod', 'FirstTopDressFert', "1tdUrea",'1tdUrea_per_Acre', '1appDaysUrea', "2tdUrea", '2tdUrea_per_Acre', '2appDaysUrea', 'MineralFertAppMethod.1',
         'Harv_method', 'Harv_date', "Harv_hand_rent", 'Harv_hand_rent_per_Acre', 'Threshing_date', 'Threshing_method','Residue_length', 'Residue_perc', 'Stubble_use',
         'Acre', 'Yield','Yield_per_Acre'
         ]]


# #### Parsing messy categorical variables

# In[142]:


# PARSING MESSY CATEGORICAL VARIABLES

# 1. LandPreparationMethod
#methods = ["TractorPlough","FourWheelTracRotavator","WetTillagePuddling","BullockPlough","Other"]

df["LandPrepMethod_TractorPlough"] = df["LandPreparationMethod"].str.contains("TractorPlough")
df["LandPrepMethod_FourWheelTracRotavator"] = df["LandPreparationMethod"].str.contains("FourWheelTracRotavator")
df["LandPrepMethod_WetTillagePuddling"] = df["LandPreparationMethod"].str.contains("WetTillagePuddling")
df["LandPrepMethod_BullockPlough"] = df["LandPreparationMethod"].str.contains("BullockPlough")
df["LandPrepMethod_Other"] = df["LandPreparationMethod"].str.contains("Other")


# 2. NursDetFactor
#reasons = ["CalendarDate","PreMonsoonShowers","IrrigWaterAvailability","LabourAvailability","SeedAvailability"]

df["NursDetFactor_CalendarDate"] = df["NursDetFactor"].str.contains("CalendarDate")
df["NursDetFactor_PreMonsoonShowers"] = df["NursDetFactor"].str.contains("PreMonsoonShowers")
df["NursDetFactor_IrrigWaterAvailability"] = df["NursDetFactor"].str.contains("IrrigWaterAvailability")
df["NursDetFactor_LabourAvailability"] = df["NursDetFactor"].str.contains("LabourAvailability" or "LaborAvailability")
df["NursDetFactor_SeedAvailability"] = df["NursDetFactor"].str.contains("SeedAvailability")


# 2. TransDetFactor
#reasons = ["LaborAvailability","CalendarDate","RainArrival","IrrigWaterAvailability","SeedlingAge"] # I think that's all of them

df["TransDetFactor_LabourAvailability"] = df["TransDetFactor"].str.contains("LabourAvailability" or "LaborAvailability")
df["TransDetFactor_CalendarDate"] = df["TransDetFactor"].str.contains("CalendarDate")
df["TransDetFactor_RainArrival"] = df["TransDetFactor"].str.contains("RainArrival")
df["TransDetFactor_IrrigWaterAvailability"] = df["TransDetFactor"].str.contains("IrrigWaterAvailability")
df["TransDetFactor_SeedlingAge"] = df["TransDetFactor"].str.contains("SeedlingAge")


# 3. CropbasalFerts
df["CropbasalFerts"] = df["CropbasalFerts"].fillna("None")
fertilizer_types = ["Urea","DAP","Other","NPK","MoP","NPKS","SSP","None"]

for fertilizer in fertilizer_types:
  label = "CropbasalFerts_" + fertilizer
  df[label] = df["CropbasalFerts"].str.contains(fertilizer)


# 4. FirstTopDressFert
df["FirstTopDressFert"] = df["FirstTopDressFert"].fillna("None")
fertilizer_types2 = ["Urea","DAP","NPK","NPKS","SSP","Other"]

for fertilizer in fertilizer_types2:
  label = "FirstTopDressFert_" + fertilizer
  df[label] = df["FirstTopDressFert"].str.contains(fertilizer)


# 5. OrgFertilizers
df["OrgFertilizers"] = df["OrgFertilizers"].fillna("None")
orgfertilizers = ["Ganaura","FYM","VermiCompost","Pranamrit","Ghanajeevamrit","Jeevamrit","PoultryManure"]
for fertilizer in orgfertilizers:
  label = "OrgFertilizers_" + fertilizer
  df[label] = df["OrgFertilizers"].str.contains(fertilizer)


# 6. Replacing all NaNs with False
cols = ['LandPrepMethod_TractorPlough','LandPrepMethod_FourWheelTracRotavator','LandPrepMethod_WetTillagePuddling', 'LandPrepMethod_BullockPlough','LandPrepMethod_Other',
        'NursDetFactor_CalendarDate','NursDetFactor_PreMonsoonShowers','NursDetFactor_IrrigWaterAvailability','NursDetFactor_LabourAvailability', 'NursDetFactor_SeedAvailability',
        'TransDetFactor_LabourAvailability', 'TransDetFactor_CalendarDate','TransDetFactor_RainArrival', 'TransDetFactor_IrrigWaterAvailability','TransDetFactor_SeedlingAge',
        'CropbasalFerts_Urea','CropbasalFerts_DAP', 'CropbasalFerts_Other', 'CropbasalFerts_NPK','CropbasalFerts_MoP', 'CropbasalFerts_NPKS', 'CropbasalFerts_SSP',
        'CropbasalFerts_None', 'FirstTopDressFert_Urea','FirstTopDressFert_DAP', 'FirstTopDressFert_NPK','FirstTopDressFert_NPKS', 'FirstTopDressFert_SSP','FirstTopDressFert_Other',
        'OrgFertilizers_Ganaura','OrgFertilizers_FYM', 'OrgFertilizers_VermiCompost','OrgFertilizers_Pranamrit', 'OrgFertilizers_Ghanajeevamrit','OrgFertilizers_Jeevamrit',
        'OrgFertilizers_PoultryManure']

for col in cols:
  df[col] = df[col].fillna(False)


# #### Missing values

# In[143]:


# For 1appDaysUrea, 2appDaysUrea, in most cases NaN means there was so 2nd or 3rd dose, so NaN is appropriate.
# For a couple rows, however, there was a 2nd or 3rd dose (as indicated by 1tdUrea and 2tdUrea) but the number of days value is missing, in which case they probably need imputation

# For rows where XappDaysUrea is NaN but XtdUrea is not NaN, impute with block median
subset = df.loc[(df["Block"]=="Rajgir")]
df.loc[(df["1appDaysUrea"].isnull()==True) & (df["1tdUrea"].isnull()==False), "1appDaysUrea"] = subset["1tdUrea"].median()

subset = df.loc[(df["Block"]=="Gurua")]
df.loc[(df["2appDaysUrea"].isnull()==True) & (df["2tdUrea"].isnull()==False), "2appDaysUrea"] = subset["2tdUrea"].median()

# Imputing with full sample medians
#df.loc[(df["1appDaysUrea"].isnull()==True) & (df["1tdUrea"]!=0), "1appDaysUrea"] = df["1appDaysUrea"].median()
#df.loc[(df["2appDaysUrea"].isnull()==True) & (df["2tdUrea"]!=0), "2appDaysUrea"] = df["2appDaysUrea"].median()


# In[144]:


# Replacing NaN with 0 for columns where it makes sense
fillna0 = ["2tdUrea","1tdUrea","Harv_hand_rent","Ganaura","CropOrgFYM","BasalDAP","BasalUrea"]
for col in fillna0:
  df[col] = df[col].fillna(0)

# Creating a new variable counting the number of missing values for each row (excluding outcome variables)
df["Nb_of_NaN"] = df.drop(columns=["Yield","Yield_per_Acre"]).isnull().sum(axis=1)
# in case that's useful to define a threshold and drop rows that are too incomplete
print(df.loc[df["Nb_of_NaN"]>10].shape[0], "rows have over 10 missing values")
# updated thought: a bunch of NaNs are actually meaningful, so I don't think defining a threshold is useful


# In[145]:


# For TransplantingIrrigationHours, TransplantingIrrigationSource, and TransplantingIrrigationPowerSource; no significant statistical difference between NaNs and non-NaNs on yields.

print("TransplantingIrrigationSource - Number of NaNs before imputation: ", df["TransplantingIrrigationSource"].isnull().sum())
print("TransplantingIrrigationPowerSource - Number of NaNs before imputation: ", df["TransplantingIrrigationPowerSource"].isnull().sum())
print("TransplantingIrrigationHours - Number of NaNs before imputation: ", df["TransplantingIrrigationHours"].isnull().sum())

# For TransplantingIrrigationSource and TransplantingIrrigationPowerSource, imputing missing values with most common category (mode)
df.loc[df["TransplantingIrrigationSource"].isnull()==True, "TransplantingIrrigationSource"] = df["TransplantingIrrigationSource"].mode()
df.loc[df["TransplantingIrrigationPowerSource"].isnull()==True, "TransplantingIrrigationPowerSource"] = df["TransplantingIrrigationPowerSource"].mode()

# For TransplantingIrrigationHours, imputing missing values with the median
df.loc[df["TransplantingIrrigationHours"].isnull()==True, "TransplantingIrrigationHours"] = df["TransplantingIrrigationHours"].median()


# Remaining variables with NaNs:
# - RcNursEstDate: statistically significant difference between rows with missing values vs. not in this variable, so leaving NaNs alone for now
# - SeedlingsPerPit: same thing
# - NursDetFactor: same thing
# - TransIrriCost: same thing
# - StandingWater: same thing

# #### Basic feature engineering

# 1. Creating variables for the number of days between different steps

# In[146]:


df.select_dtypes(include="datetime")


# In[147]:


# Manually correcting 3 rows that have input errors in Harv_date
df.loc[df["Harv_date"]=="2021-12-01", "Harv_date"] = "2022-12-01" # Wrong year
df.loc[df["Harv_date"]=="2021-12-03", "Harv_date"] = "2022-12-03" # Wrong year
df.loc[df["Harv_date"]=="2022-03-04", "Harv_date"] = "2022-11-04" # I'm guessing the month is messed up, should probably be November


# In[148]:


# Number of days between Nursing and Sowing/Transplanting
df["Days_bw_Nurs_SowTransp"] = df["SeedingSowingTransplanting"] - df["RcNursEstDate"]

# Number of days between Sowing/Transplanting and Harvesting
df["Days_bw_SowTransp_Harv"] = df["Harv_date"] - df["SeedingSowingTransplanting"]

# Number of days between Harvesting and Threshing
df["Days_bw_Harv_Thresh"] = df["Threshing_date"] - df["Harv_date"]


# In[149]:


# Re-formatting the new variables
days_cols = ["Days_bw_Nurs_SowTransp","Days_bw_SowTransp_Harv","Days_bw_Harv_Thresh"]

for col in days_cols:
  df[col] = df[col].astype(str).str[:-5]

# for rows where RcNursEstDate = NaN:
df.loc[df["Days_bw_Nurs_SowTransp"]=='', "Days_bw_Nurs_SowTransp"] = np.nan

for col in days_cols:
  df[col] = df[col].astype(float)


# In[150]:


# Visualizing these new variables
fig=px.box(df, x="Days_bw_Nurs_SowTransp", width=600, height=350, labels={'Days_bw_Nurs_SowTransp':'Number of days between Nursing and Sowing/Transplanting'})
fig.show()

fig=px.box(df, x="Days_bw_SowTransp_Harv", width=600, height=350, labels={'Days_bw_SowTransp_Harv':'Number of days between Sowing/Transplanting and Harvesting'})
fig.show()

fig=px.box(df, x="Days_bw_Harv_Thresh", width=600, height=350, labels={'Days_bw_Harv_Thresh':'Number of days between Harvesting and Threshing'})
fig.show()


# 2. Extracting months from date variables

# In[151]:


# Extracting months from date variables
df["CropTillageMonth"] = df["CropTillageDate"].dt.month_name()
df["NursingMonth"] = df["RcNursEstDate"].dt.month_name()
df["SowTransplantMonth"] = df["SeedingSowingTransplanting"].dt.month_name()
df["HarvestMonth"] = df["Harv_date"].dt.month_name()
df["ThreshingMonth"] = df["Threshing_date"].dt.month_name()


# In[152]:


df["HarvestMonth"].value_counts()


# #### Exporting cleaned df

# In[153]:


# Dropping messy cols
df = df.drop(columns=["LandPreparationMethod","NursDetFactor","TransDetFactor","OrgFertilizers","CropbasalFerts","FirstTopDressFert"])


# In[154]:


# Re-ordering columns for clarity
df = df[['ID','Set','District','Block','CropTillageDate','CropTillageMonth','CropTillageDepth','CropEstMethod','RcNursEstDate','NursingMonth',
         'SeedingSowingTransplanting','SowTransplantMonth','SeedlingsPerPit','TransplantingIrrigationHours','TransplantingIrrigationSource',
         'TransplantingIrrigationPowerSource','TransIrriCost','TransIrriCost_per_Acre','StandingWater','Ganaura','Ganaura_per_Acre','CropOrgFYM',
         'CropOrgFYM_per_Acre','PCropSolidOrgFertAppMethod','NoFertilizerAppln','BasalDAP','BasalDAP_per_Acre','BasalUrea','BasalUrea_per_Acre',
         'MineralFertAppMethod','1tdUrea','1tdUrea_per_Acre','1appDaysUrea','2tdUrea','2tdUrea_per_Acre','2appDaysUrea','MineralFertAppMethod.1',
         'Harv_method','Harv_date','HarvestMonth','Harv_hand_rent','Harv_hand_rent_per_Acre','Threshing_date','ThreshingMonth','Threshing_method',
         'Residue_length','Residue_perc','Stubble_use','Acre','Yield','Yield_per_Acre',
         # Parsed categoricals
         'LandPrepMethod_TractorPlough','LandPrepMethod_FourWheelTracRotavator','LandPrepMethod_WetTillagePuddling',
         'LandPrepMethod_BullockPlough','LandPrepMethod_Other','NursDetFactor_CalendarDate','NursDetFactor_PreMonsoonShowers','NursDetFactor_IrrigWaterAvailability',
         'NursDetFactor_LabourAvailability','NursDetFactor_SeedAvailability','TransDetFactor_LabourAvailability','TransDetFactor_CalendarDate','TransDetFactor_RainArrival',
         'TransDetFactor_IrrigWaterAvailability','TransDetFactor_SeedlingAge','CropbasalFerts_Urea','CropbasalFerts_DAP','CropbasalFerts_Other','CropbasalFerts_NPK',
         'CropbasalFerts_MoP','CropbasalFerts_NPKS','CropbasalFerts_SSP','CropbasalFerts_None','FirstTopDressFert_Urea','FirstTopDressFert_DAP','FirstTopDressFert_NPK',
         'FirstTopDressFert_NPKS','FirstTopDressFert_SSP','FirstTopDressFert_Other','OrgFertilizers_Ganaura','OrgFertilizers_FYM','OrgFertilizers_VermiCompost',
         'OrgFertilizers_Pranamrit','OrgFertilizers_Ghanajeevamrit','OrgFertilizers_Jeevamrit','OrgFertilizers_PoultryManure','Days_bw_Nurs_SowTransp',
         'Days_bw_SowTransp_Harv','Days_bw_Harv_Thresh',
         # Other additions
         'Nb_of_NaN'
         ]]


# In[155]:


# EXPORTING

# V1: full cleaned df
df.to_csv('cleaned_fulldf.csv',index=False)

# V2: per Acre df (dropping the raw variables)
#df.copy().drop(columns=["TransIrriCost","Ganaura","CropOrgFYM","BasalDAP","BasalUrea","1tdUrea","2tdUrea","Harv_hand_rent","Yield"]).to_csv('peracre_df.csv',index=False)

# V2: per Acre df (dropping the raw variables)
#df.copy().drop(columns=["TransIrriCost_per_Acre","Ganaura_per_Acre","CropOrgFYM_per_Acre","BasalDAP_per_Acre","BasalUrea_per_Acre","1tdUrea_per_Acre","2tdUrea_per_Acre",
#                        "Harv_hand_rent_per_Acre","Yield_per_Acre"]).to_csv('rawyield_df.csv',index=False)


# In[156]:


df.shape

