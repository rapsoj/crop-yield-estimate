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




