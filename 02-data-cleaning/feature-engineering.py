import pandas as pd
import numpy as np
import calendar
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

def get_date_distance(df):
  # Calculate number of days between Nursing and Sowing/Transplanting
  df["Days_bw_Nurs_SowTransp"] = df["SeedingSowingTransplanting"] - df["RcNursEstDate"]

  # Calculate number of days between Sowing/Transplanting and Harvesting
  df["Days_bw_SowTransp_Harv"] = df["Harv_date"] - df["SeedingSowingTransplanting"]

  # Calcualte number of days between Harvesting and Threshing
  df["Days_bw_Harv_Thresh"] = df["Threshing_date"] - df["Harv_date"]

  # Re-format new variables
  days_cols = ["Days_bw_Nurs_SowTransp","Days_bw_SowTransp_Harv","Days_bw_Harv_Thresh"]
  for col in days_cols:
    df[col] = df[col].astype(str).str[:-5]

  # Format new missing values
  df.loc[df["Days_bw_Nurs_SowTransp"]=='', "Days_bw_Nurs_SowTransp"] = np.nan

  # Format date distances as floats
  for col in days_cols:
    df[col] = df[col].astype(float)

  return(df)


def get_months(df):
  # Extract months from date variables
  df["CropTillageMonth"] = df["CropTillageDate"].dt.month_name()
  df["NursingMonth"] = df["RcNursEstDate"].dt.month_name()
  df["SowTransplantMonth"] = df["SeedingSowingTransplanting"].dt.month_name()
  df["HarvestMonth"] = df["Harv_date"].dt.month_name()
  df["ThreshingMonth"] = df["Threshing_date"].dt.month_name()

  return df





# In[ ]:


# Create feature for past month yield per acre 'Past_Yield_per_Acre'
train['Harv_date'] = pd.to_datetime(train['Harv_date'])
train.sort_values(['District', 'Harv_date'], inplace=True)

# Group the DataFrame by 'District' and calculate the rolling average
train['Past_YpA_Avg'] = train.groupby('District')['Yield_per_Acre'].rolling(
    window = 30).mean().reset_index(0, drop=True)

# Fill NaN values in the 'past_month_avg' column with 0 if needed
train['Past_YpA_Avg'].fillna(0, inplace=True)


# In[ ]:


# Create feature for days between harvesting and threshing 'Days_Harv_Thresh'
train['Threshing_date'] = pd.to_datetime(train['Threshing_date'])
train['Days_Harv_Thresh'] = (
    train['Threshing_date'] - train['Harv_date']).dt.days





#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import calendar
from datetime import datetime
pd.set_option('display.max_columns', None)

from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")


# In[9]:


training_data = pd.read_csv('T:\crop-yield-estimate\data\Train.csv') # Replace with your own path


# In[10]:


training_data.head()


# In[11]:


data = pd.read_csv('cleaned_fulldf_withclusters.csv')


# In[12]:


data.head()


# In[17]:


# Function to assign season based on month names for India's climatic pattern
def assign_season(month_name):
    if month_name in ['June', 'July', 'August', 'September']:  # Monsoon season
        return 'Monsoon'
    elif month_name in ['October', 'November', 'December']:  # Post-Monsoon season
        return 'Post-Monsoon'
    elif month_name in ['January', 'February', 'March']:  # Winter season
        return 'Winter'
    elif month_name in ['April', 'May']:  # Summer season
        return 'Summer'


# In[18]:




# Apply the function to assign season
data['Season'] = data['CropTillageMonth'].apply(assign_season)

data[['CropTillageDate', 'Season']].head()  # Displaying the first few rows for verification


# In[19]:


data['Season'].value_counts()  # Count of each season


# In[20]:


data.head()


# In[21]:


data['Total_Crop_Cycle_Duration'] = data['Days_bw_Nurs_SowTransp'] + data['Days_bw_SowTransp_Harv'] + data['Days_bw_Harv_Thresh']


# In[22]:


data['Total_Crop_Cycle_Duration'].describe()


# In[29]:


data.groupby('k3label')['Total_Crop_Cycle_Duration'].mean()


# In[30]:


data = data.drop('Nb_of_NaN', axis=1)


# In[38]:




# View all rows and columns
pd.set_option('display.max_rows', None)


# In[39]:


# Where do we still have NaNs?
data.isna().sum()


# Strategy for Each one:
# 
# 1. Seedlings per pit, use K nearest to extraploate
# 2. RcNursEstDate, Exploring

# In[40]:


data[data['RcNursEstDate'].isna()]


# In[34]:


# Where there is no NaN for any fertiliser related columns, we can replace NaN with 0

data['BasalDAP_per_Acre'] = data['BasalDAP_per_Acre'].fillna(0)


# In[ ]:







def reorder_cols(df):
  # Re-order columns for clarity
  df = df[['ID','Set','District','Block','CropTillageDate','CropTillageMonth','CropTillageDepth','CropEstMethod',
          'RcNursEstDate','NursingMonth', 'SeedingSowingTransplanting','SowTransplantMonth','SeedlingsPerPit',
          'TransplantingIrrigationHours','TransplantingIrrigationSource', 'TransplantingIrrigationPowerSource',
          'TransIrriCost','TransIrriCost_per_Acre','StandingWater','Ganaura','Ganaura_per_Acre','CropOrgFYM', 
          'CropOrgFYM_per_Acre','PCropSolidOrgFertAppMethod','NoFertilizerAppln','BasalDAP','BasalDAP_per_Acre',
          'BasalUrea','BasalUrea_per_Acre','MineralFertAppMethod','1tdUrea','1tdUrea_per_Acre','1appDaysUrea',
          '2tdUrea','2tdUrea_per_Acre','2appDaysUrea','MineralFertAppMethod.1','Harv_method','Harv_date',
          'HarvestMonth','Harv_hand_rent','Harv_hand_rent_per_Acre','Threshing_date','ThreshingMonth',
          'Threshing_method','Residue_length','Residue_perc','Stubble_use','Acre','Yield','Yield_per_Acre',
          # Parsed categoricals
          'LandPrepMethod_TractorPlough','LandPrepMethod_FourWheelTracRotavator','LandPrepMethod_WetTillagePuddling',
          'LandPrepMethod_BullockPlough','LandPrepMethod_Other','NursDetFactor_CalendarDate',
          'NursDetFactor_PreMonsoonShowers','NursDetFactor_IrrigWaterAvailability',
          'NursDetFactor_LabourAvailability','NursDetFactor_SeedAvailability','TransDetFactor_LabourAvailability',
          'TransDetFactor_CalendarDate','TransDetFactor_RainArrival','TransDetFactor_IrrigWaterAvailability',
          'TransDetFactor_SeedlingAge','CropbasalFerts_Urea','CropbasalFerts_DAP','CropbasalFerts_Other',
          'CropbasalFerts_NPK','CropbasalFerts_MoP','CropbasalFerts_NPKS','CropbasalFerts_SSP',
          'CropbasalFerts_None','FirstTopDressFert_Urea','FirstTopDressFert_DAP','FirstTopDressFert_NPK',
          'FirstTopDressFert_NPKS','FirstTopDressFert_SSP','FirstTopDressFert_Other','OrgFertilizers_Ganaura',
          'OrgFertilizers_FYM','OrgFertilizers_VermiCompost','OrgFertilizers_Pranamrit','OrgFertilizers_Ghanajeevamrit',
          'OrgFertilizers_Jeevamrit','OrgFertilizers_PoultryManure','Days_bw_Nurs_SowTransp',
          'Days_bw_SowTransp_Harv','Days_bw_Harv_Thresh',
          # Other additions
          'Nb_of_NaN']]

  return df


def get_features(train_path, test_path):
  df = load_data(train_path, test_path)
  df = get_date_distance(df)
  df = get_months(df)

  return df