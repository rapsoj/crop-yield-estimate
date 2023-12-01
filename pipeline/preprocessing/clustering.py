import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")

def run_kmeans(k_range, input_df, output_df, var_name):
    col_list = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(input_df)
        colname = var_name + '_' + f'k{k}_label'
        col_list.append(colname)
        output_df[colname] = kmeans.labels_
    return output_df, col_list


def get_kmeans_all(df):
  # Calculate k-means clusters for all variables
  cdf = df[['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11',
            'PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19','PC20','PC21']]

  # Run k-means clustering
  var_name = 'all'
  k_range = range(2,6)
  cdf_results = cdf.copy()
  run_kmeans(k_range, cdf, cdf_results, var_name)

  # Add results to data
  data = pd.get_dummies(cdf_results.iloc[:,-k_range[-2]:], columns = list(cdf_results.iloc[:,-k_range[-2]:].columns), drop_first=True)
  df = pd.concat([df, data], axis=1)

  return df


def get_kmeans_crop(df):
  # Calculate k-means clusters for crop identification variables
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
          'MineralFertAppMethod_2_SoilApplied','Harv_method_machine','Threshing_method_machine',
          'Days_bw_Nurs_SowTransp_ModeDiff_Imputed','Days_bw_Nurs_Harv_ModeDiff_Imputed',
          'Days_bw_Nurs_Till_ModeDiff_Imputed','Days_bw_Till_SowTransp_ModeDiff',
          'Days_bw_Till_Harv_ModeDiff','Days_bw_SowTransp_Harv_ModeDiff',
          'Days_bw_Harv_Thresh_ModeDiff','HarvestDate_ModeDiff','ThreshingDate_ModeDiff',
          'NursingDate_ModeDiff_Imputed','TillageDate_ModeDiff','SowTransplantDate_ModeDiff']]

  # Run k-means clustering
  var_name = 'crop_id'
  k_range = range(2,6)
  cdf_results = cdf.copy()
  run_kmeans(k_range, cdf, cdf_results, var_name)

  # Add results to data
  data = pd.get_dummies(cdf_results.iloc[:,-k_range[-2]:], columns = list(cdf_results.iloc[:,-k_range[-2]:].columns), drop_first=True)
  df = pd.concat([df, data], axis=1)

  return df


def get_kmeans_important(df):
  # Calculate k-means clusters for most important variables
  cdf = df[['2appDaysUrea_Imputed','TpIrrigationCost_Imputed_per_Acre','TillageDate_ModeDiff','2appDaysUrea_Imputed','PC10','PC4',
            'TpIrrigationHours_Imputed_per_Acre','2appDaysUrea_Imputed_MeanDiff','TpIrrigationHours_Imputed_per_Acre','PC6',
            'Ganaura_capped','PC15','PCropSolidOrgFertAppMethod_Broadcasting','TpIrrigationCost_Imputed_per_Acre_capped',
            '2appDaysUrea_Imputed_MeanDiff','Days_bw_SowTransp_Harv','Num_TopDressFert','MineralFertAppMethod_1_SoilApplied',
            'CropOrgFYM_per_Acre','CropTillageDepth','PC9','Ganaura','SowTransplantDate_ModeDiff','Num_NursDetFactor',
            'CropbasalFerts_SSP_True','Num_TransDetFactor','1appDaysUrea_Imputed','HarvestDate_ModeDiff','PC17',
            'Days_bw_Nurs_SowTransp_ModeDiff_Imputed']]

  # Run k-means clustering
  var_name = 'top_shapley'
  k_range = range(2,6)
  cdf_results = cdf.copy()
  run_kmeans(k_range, cdf, cdf_results, var_name)

  # Add results to data
  data = pd.get_dummies(cdf_results.iloc[:,-k_range[-2]:], columns = list(cdf_results.iloc[:,-k_range[-2]:].columns), drop_first=True)
  df = pd.concat([df, data], axis=1)

  return df


def get_clusters(df):
  df = get_kmeans_all(df)
  df = get_kmeans_crop(df)
  df = get_kmeans_important(df)

  return df