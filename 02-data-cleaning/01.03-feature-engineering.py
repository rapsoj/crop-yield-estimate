#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/rapsoj/crop-yield-estimate/blob/main/01.02-feature-engineering.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# #01.02 Feature Engineering
# Creating new features to improve predictions for the Digital Green Crop Yield Estimate Challenge.

# ### Prepare Workspace

# In[2]:


# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# Import data manipulation libraries
import pandas as pd
import numpy as np


# In[ ]:


# Load files
data_path = '/content/drive/MyDrive/Colab Notebooks/crop-yield-estimate/'
train = pd.read_csv(data_path + 'Train.csv')
test = pd.read_csv(data_path + 'Test.csv')
var_desc = pd.read_csv(data_path + 'VariableDescription.csv')


# ### Prepare Workspace Locally

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


data_path = '/crop-yield-estimate/data/'
train = pd.read_csv(data_path + 'Train.csv')
test = pd.read_csv(data_path + 'Test.csv')
var_desc = pd.read_csv(data_path + 'VariableDescription.csv')


# In[5]:


train.head()


# In[6]:


count = len(train[train['CultLand'] < train['CropCultLand']])
print(count)


# In[7]:


count = len(train[train['CultLand'] > train['CropCultLand']])
print(count)


# In[9]:



# Calculate the correlation between CultLand and Acre columns
corr_cultland = train['CultLand'].corr(train['Acre'])

# Calculate the correlation between CropCultLand and Acre columns
corr_cropcultland = train['CropCultLand'].corr(train['Acre'])

print(corr_cultland)
print(corr_cropcultland)


# In[ ]:


# How many missing numbers are there?
train.isnull().sum()



# In[ ]:


# Calculate percentage of missing values in each column
train.isnull().sum() / len(train) * 100


# ### Perform Feature Engineering

# In[ ]:


# Create feature for yield per acre 'Yield_per_Acre'
train['Yield_per_Acre'] = train['Yield'] / train['Acre']


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


# In[ ]:




