import pandas as pd
import numpy as np

from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")


# to add: FirstTopDressFert, CropbasalFerts, OrgFertilizers, should also add months and date differences!
cdf = df[['CropTillageDepth','CropEstMethod_LineSowingAfterTillage','CropEstMethod_Manual_PuddledLine',
          'CropEstMethod_Manual_PuddledRandom','SeedlingsPerPit_Imputed','TpIrrigationHours_Imputed',
          'TransplantingIrrigationSource_Boring','TransplantingIrrigationSource_Canal',
          'TransplantingIrrigationSource_Pond','TransplantingIrrigationSource_Rainfed',
          'LandPrepMethod_TractorPlough_True','LandPrepMethod_FourWheelTracRotavator_True',
          'LandPrepMethod_WetTillagePuddling_True','LandPrepMethod_BullockPlough_True',
          'CropbasalFerts_Urea_True','CropbasalFerts_DAP_True','CropbasalFerts_NPK_True','CropbasalFerts_NPKS_True',
          'CropbasalFerts_SSP_True','CropbasalFerts_Other_True','CropbasalFerts_None_True',
          'FirstTopDressFert_Urea_True','FirstTopDressFert_DAP_True','FirstTopDressFert_NPK_True',
          'FirstTopDressFert_None_True','OrgFertilizers_Ganaura_True','OrgFertilizers_FYM_True',
          'OrgFertilizers_VermiCompost_True','OrgFertilizers_Ghanajeevamrit_True','OrgFertilizers_None_True',
          'Ganaura_per_Acre','CropOrgFYM_per_Acre','PCropSolidOrgFertAppMethod_Broadcasting',
          'PCropSolidOrgFertAppMethod_RootApplication','PCropSolidOrgFertAppMethod_SoilApplied',
          'NoFertilizerAppln','MineralFertAppMethod_2_Broadcasting','MineralFertAppMethod_2_RootApplication',
          'MineralFertAppMethod_2_SoilApplied','Harv_method_machine','Threshing_method_machine']]



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


def get_clusters(df):

    return df