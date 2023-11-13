#!/usr/bin/env python
# coding: utf-8

# # Import & setup

# In[2]:


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


# In[ ]:


from google.colab import drive
drive.mount('/content/gdrive')

# directory
get_ipython().run_line_magic('cd', "'/content/gdrive/My Drive/Oxford/ML_for_Social_Good'")

# import
df = pd.read_csv("cleaned_fulldf.csv")
print(df.shape)


# In[ ]:


# Selecting variables indicated by Shaw -- left out date variables for now

# to add: FirstTopDressFert, CropbasalFerts, OrgFertilizers
cdf = df[["ID","CropTillageDepth","CropEstMethod","SeedlingsPerPit","TransplantingIrrigationHours","TransplantingIrrigationSource",
          # one-hot encoded LandPrepMethod
          'LandPrepMethod_TractorPlough','LandPrepMethod_FourWheelTracRotavator','LandPrepMethod_WetTillagePuddling', 'LandPrepMethod_BullockPlough','LandPrepMethod_Other',
          # one-hot encoded CropbasalFerts
          'CropbasalFerts_Urea','CropbasalFerts_DAP', 'CropbasalFerts_Other', 'CropbasalFerts_NPK','CropbasalFerts_MoP', 'CropbasalFerts_NPKS', 'CropbasalFerts_SSP',
          'CropbasalFerts_None',
          # one-hot encoded FirstTopDressFert
          'FirstTopDressFert_Urea','FirstTopDressFert_DAP', 'FirstTopDressFert_NPK','FirstTopDressFert_NPKS', 'FirstTopDressFert_SSP','FirstTopDressFert_Other',
          # one-hot encoded OrgFertilizers
          'OrgFertilizers_Ganaura','OrgFertilizers_FYM', 'OrgFertilizers_VermiCompost','OrgFertilizers_Pranamrit', 'OrgFertilizers_Ghanajeevamrit','OrgFertilizers_Jeevamrit',
          'OrgFertilizers_PoultryManure',
          #--
          "Ganaura_per_Acre","CropOrgFYM_per_Acre","PCropSolidOrgFertAppMethod","NoFertilizerAppln","MineralFertAppMethod","MineralFertAppMethod.1",
          "Harv_method","Threshing_method",#"Yield_per_Acre"
        ]]

cdf.head()


# In[ ]:


cdf.columns


# # Pre-processing

# In[ ]:


# 1. CATEGORICAL VARIABLES

# Binary variables
cdf["Harv_method"] = cdf["Harv_method"].replace({"hand":0, "machine":1})
cdf["Threshing_method"] = cdf["Threshing_method"].replace({"hand":0, "machine":1})

# Dummies
dummy_cols = ["CropEstMethod","TransplantingIrrigationSource","PCropSolidOrgFertAppMethod","MineralFertAppMethod","MineralFertAppMethod.1"]
cdf = pd.get_dummies(cdf, columns=dummy_cols)

# Bool -> int
bools = cdf.select_dtypes(include='bool').columns
cdf[bools] = cdf[bools].astype(int)


# In[ ]:


# 2. MISSING DATA

# Ganaura_per_Acre & CropOrgFYM_per_Acre -> replacing with 0
cdf["Ganaura_per_Acre"] = cdf["Ganaura_per_Acre"].fillna(0)
cdf["CropOrgFYM_per_Acre"] = cdf["CropOrgFYM_per_Acre"].fillna(0)

# SeedlingsPerPit -> replacing by median (=2)
cdf["SeedlingsPerPit"] = cdf["SeedlingsPerPit"].fillna(cdf.SeedlingsPerPit.median())

# TransplantingIrrigationHours -> replacing by median (=4)
cdf["TransplantingIrrigationHours"] = cdf["TransplantingIrrigationHours"].fillna(cdf.TransplantingIrrigationHours.median())


# In[ ]:


# 3. NUMERICAL VARIABLES

num_cols = ["CropTillageDepth","SeedlingsPerPit","TransplantingIrrigationHours","NoFertilizerAppln","CropOrgFYM_per_Acre","Ganaura_per_Acre"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cdf[num_cols])

# Normalizing
cdf[num_cols] = normalize(X_scaled)


# In[ ]:


cdf = cdf.drop(columns=["ID"])


# # Spectral clustering

# In[ ]:


cdf


# In[ ]:


def run_spectral(k_range, input_df, output_df):
  for k in k_range:
    spectral = SpectralClustering(n_clusters=k, assign_labels='discretize', random_state=0, n_init=200).fit(input_df)
    colname = 'k' + str(k) + "label"
    col_list.append(colname)
    output_df[colname] = spectral.labels_


# In[ ]:


col_list = []
k_range = range(2,6)
cdf_results = cdf.copy()

run_spectral(k_range, cdf, cdf_results)


# In[ ]:


print(cdf_results["k2label"].value_counts())
print(cdf_results["k3label"].value_counts())
print(cdf_results["k4label"].value_counts())
print(cdf_results["k5label"].value_counts())


# In[ ]:


cdf_results["k2label"] = cdf_results["k2label"].replace({0:"A", 1:"B"})
cdf_results["k3label"] = cdf_results["k3label"].replace({0:"A", 1:"B", 2:"C"})
cdf_results["k4label"] = cdf_results["k4label"].replace({0:"A", 1:"B", 2:"C", 3:"D"})
cdf_results["k5label"] = cdf_results["k5label"].replace({0:"A", 2:"B", 1:"C", 3:"D", 4:"E"})


# In[ ]:


cdf_results.loc[cdf_results["k5label"]=="A"].k2label.value_counts()


# In[ ]:


# MERGING CLUSTER LABELS BACK WITH ORIGINAL DF

tempo = cdf_results[["k2label","k3label","k4label","k5label"]]
df2 = pd.concat([df,tempo], axis=1)
tempo.shape, df.shape, df2.shape


# In[ ]:


df2.loc[df2["k2label"]=="B"].CropTillageDepth.value_counts()


# In[ ]:


fig = px.box(df2, x="CropTillageDepth", color="k3label", width=600, height=400)
fig.show()

fig = px.box(df2, x="SeedlingsPerPit", color="k3label", width=600, height=400)
fig.show()

fig = px.histogram(df2, x="TransplantingIrrigationSource", color="k3label", width=600, height=400)
fig.show()

#fig = px.histogram(df2, x="OrgFertilizers", color="k3label", width=600, height=400)
#fig.show()

fig = px.box(df2, x="CropOrgFYM_per_Acre", color="k3label", width=600, height=400)
fig.show()

fig = px.histogram(df2, x="Harv_method", facet_col="k3label", width=600, height=400)
fig.show()

fig = px.histogram(df2, x="Threshing_method", facet_col="k3label", width=600, height=400)
fig.show()

fig = px.histogram(df2, x="District", facet_col="k3label", width=800, height=400)
fig.show()

fig = px.histogram(df2, x="Block", facet_col="k3label", width=900, height=400)
fig.show()

fig = px.box(df2, x="Yield_per_Acre", color="k3label", width=900, height=400)
fig.show()

fig = px.box(df2, x="Yield_per_Acre", color="k2label", width=900, height=400)
fig.show()


# In[ ]:


# Exporting df with cluster labels
df2.to_csv('cleaned_fulldf_withclusters.csv',index=False)

