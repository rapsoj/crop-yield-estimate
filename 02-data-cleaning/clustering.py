import pandas as pd
import numpy as np
import calendar
from datetime import datetime

from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")

# to add: FirstTopDressFert, CropbasalFerts, OrgFertilizers, should also add months and date differences!
cdf = df[["ID","CropTillageDepth","CropEstMethod","SeedlingsPerPit","TransplantingIrrigationHours","TransplantingIrrigationSource",
          # Add one-hot encoded LandPrepMethod
          'LandPrepMethod_TractorPlough','LandPrepMethod_FourWheelTracRotavator','LandPrepMethod_WetTillagePuddling', 'LandPrepMethod_BullockPlough','LandPrepMethod_Other',
          # Addone-hot encoded CropbasalFerts
          'CropbasalFerts_Urea','CropbasalFerts_DAP', 'CropbasalFerts_Other', 'CropbasalFerts_NPK','CropbasalFerts_MoP', 'CropbasalFerts_NPKS', 'CropbasalFerts_SSP',
          'CropbasalFerts_None',
          # Add one-hot encoded FirstTopDressFert
          'FirstTopDressFert_Urea','FirstTopDressFert_DAP', 'FirstTopDressFert_NPK','FirstTopDressFert_NPKS', 'FirstTopDressFert_SSP','FirstTopDressFert_Other',
          # Add one-hot encoded OrgFertilizers
          'OrgFertilizers_Ganaura','OrgFertilizers_FYM', 'OrgFertilizers_VermiCompost','OrgFertilizers_Pranamrit', 'OrgFertilizers_Ghanajeevamrit','OrgFertilizers_Jeevamrit',
          'OrgFertilizers_PoultryManure',
          # Add other variables
          "Ganaura_per_Acre","CropOrgFYM_per_Acre","PCropSolidOrgFertAppMethod","NoFertilizerAppln","MineralFertAppMethod","MineralFertAppMethod.1",
          "Harv_method","Threshing_method"]]




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




def make_clusters(df):
  # Create clusters for 
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


