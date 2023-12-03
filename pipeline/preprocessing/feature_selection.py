import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def regroup_test_only_col(df):
  # Regroup column that only appears in the test set
  df['MineralFertAppMethod_1_SoilApplied'] = np.where(df['MineralFertAppMethod_1_Spray'], True, df['MineralFertAppMethod_1_SoilApplied'])

  return df


def drop_unusable_vars(df):
  df = df.drop(columns = ['Residuals','ID','LandPreparationMethod','CropTillageDate','RcNursEstDate','SeedingSowingTransplanting',
                          'NursDetFactor','TransDetFactor','OrgFertilizers','CropbasalFerts','FirstTopDressFert','Harv_date','Threshing_date',
                          'LandPrepMethod_Other','CropbasalFerts_MoP','FirstTopDressFert_NPKS','FirstTopDressFert_SSP',
                          'FirstTopDressFert_Other','OrgFertilizers_Pranamrit','OrgFertilizers_Jeevamrit','OrgFertilizers_PoultryManure',
                          'CropTillageSeason','NursingSeason','SowTransplantSeason','HarvestSeason','MineralFertAppMethod_1_Spray'])

  return df


def drop_low_values(df):
  # Drop one-hot encoded columns with less than 20 values 
  cols_to_drop = ['Block_Gaya','TpIrrigationSource_Imputed_Well','TpIrrigationPowerSource_Imputed_Solar',
                  'TransplantingIrrigationSource_Well','TransplantingIrrigationPowerSource_Solar','TpIrrigationSource_Imputed_TubeWell',
                  'TransplantingIrrigationSource_TubeWell','MineralFertAppMethod_1_RootApplication','PCropSolidOrgFertAppMethod_Spray']
  cols_to_drop_existing = [col for col in cols_to_drop if col in df.columns]
  df = df.drop(columns = cols_to_drop_existing)

  return df


def drop_test_only_vars(df):
  # Drop variables that appear only in the test set
  cols_to_drop = ['Block_Lohra']
  cols_to_drop_existing = [col for col in cols_to_drop if col in df.columns]
  df = df.drop(columns = cols_to_drop_existing)

  return df


def select_features(df):
    df = regroup_test_only_col(df)
    df = drop_unusable_vars(df)
    df = drop_low_values(df)
    df = drop_test_only_vars(df)

    return df