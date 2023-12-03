import pandas as pd
import numpy as np
import calendar
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")


def load_data(train_path, test_path):
  df_train = pd.read_csv(train_path)
  df_test = pd.read_csv(test_path)

  # Add column for yield to test dataframe
  df_test['Yield'] = np.nan

  # Combine data
  df = pd.concat([df_train, df_test], axis=0)
     
  return df


def fix_duplicate(df):

  df = df.rename(columns={'MineralFertAppMethod': 'MineralFertAppMethod_1','MineralFertAppMethod.1': 'MineralFertAppMethod_2'})

  return df


def adjust_datetime_columns(df):
  datetime_cols = ["CropTillageDate", "RcNursEstDate", "SeedingSowingTransplanting", "Harv_date", "Threshing_date"]
  for col in datetime_cols:
    df[col] = pd.to_datetime(df[col])

  return df


def fix_errors(df):
  # Fix district typo
  typo_row = df[(df["District"] == "Jamui") & (df["Block"] == "Gurua")].index
  df.loc[typo_row, "Block"] = "Gaya"

  # For rows where XappDaysUrea is NaN but XtdUrea is not NaN, impute with block median
  subset = df.loc[(df["Block"]=="Rajgir")]
  df.loc[(df["1appDaysUrea"].isnull()==True) & (df["1tdUrea"].isnull()==False), "1appDaysUrea"] = subset["1tdUrea"].median()
  subset = df.loc[(df["Block"]=="Gurua")]
  df.loc[(df["2appDaysUrea"].isnull()==True) & (df["2tdUrea"].isnull()==False), "2appDaysUrea"] = subset["2tdUrea"].median()

  # Manually correct 3 rows that have input errors in Harv_date
  df.loc[df["Harv_date"]=="2021-12-01", "Harv_date"] = "2022-12-01" # Wrong year
  df.loc[df["Harv_date"]=="2021-12-03", "Harv_date"] = "2022-12-03" # Wrong year
  df.loc[df["Harv_date"]=="2022-03-04", "Harv_date"] = "2022-11-04" # Month may be messed up, should probably be November

  return df


def parse_categorical(df):
  df["LandPrepMethod_TractorPlough"] = df["LandPreparationMethod"].str.contains("TractorPlough")
  df["LandPrepMethod_FourWheelTracRotavator"] = df["LandPreparationMethod"].str.contains("FourWheelTracRotavator")
  df["LandPrepMethod_WetTillagePuddling"] = df["LandPreparationMethod"].str.contains("WetTillagePuddling")
  df["LandPrepMethod_BullockPlough"] = df["LandPreparationMethod"].str.contains("BullockPlough")
  df["LandPrepMethod_Other"] = df["LandPreparationMethod"].str.contains("Other")

  df["NursDetFactor_CalendarDate"] = df["NursDetFactor"].str.contains("CalendarDate")
  df["NursDetFactor_PreMonsoonShowers"] = df["NursDetFactor"].str.contains("PreMonsoonShowers")
  df["NursDetFactor_IrrigWaterAvailability"] = df["NursDetFactor"].str.contains("IrrigWaterAvailability")
  df["NursDetFactor_LabourAvailability"] = df["NursDetFactor"].str.contains("LabourAvailability|LaborAvailability")
  df["NursDetFactor_SeedAvailability"] = df["NursDetFactor"].str.contains("SeedAvailability")

  df["TransDetFactor_LabourAvailability"] = df["TransDetFactor"].str.contains("LabourAvailability|LaborAvailability")
  df["TransDetFactor_CalendarDate"] = df["TransDetFactor"].str.contains("CalendarDate")
  df["TransDetFactor_RainArrival"] = df["TransDetFactor"].str.contains("RainArrival")
  df["TransDetFactor_IrrigWaterAvailability"] = df["TransDetFactor"].str.contains("IrrigWaterAvailability")
  df["TransDetFactor_SeedlingAge"] = df["TransDetFactor"].str.contains("SeedlingAge")

  df["CropbasalFerts"] = df["CropbasalFerts"].fillna("None")
  fertilizer_types = ["Urea","DAP","NPK","MoP","NPKS","SSP","Other","None"]
  for fertilizer in fertilizer_types:
    label = "CropbasalFerts_" + fertilizer
    df[label] = df["CropbasalFerts"].str.contains(fertilizer)

  df["FirstTopDressFert"] = df["FirstTopDressFert"].fillna("None")
  fertilizer_types2 = ["Urea","DAP","NPK","NPKS","SSP","Other","None"]
  for fertilizer in fertilizer_types2:
    label = "FirstTopDressFert_" + fertilizer
    df[label] = df["FirstTopDressFert"].str.contains(fertilizer)

  df["OrgFertilizers"] = df["OrgFertilizers"].fillna("None")
  orgfertilizers = ["Ganaura","FYM","VermiCompost","Pranamrit","Ghanajeevamrit","Jeevamrit","PoultryManure","None"]
  for fertilizer in orgfertilizers:
    label = "OrgFertilizers_" + fertilizer
    df[label] = df["OrgFertilizers"].str.contains(fertilizer)

  return df


def replace_binary_nan(df):
  cols = ['LandPrepMethod_TractorPlough','LandPrepMethod_FourWheelTracRotavator','LandPrepMethod_WetTillagePuddling',
          'LandPrepMethod_BullockPlough','LandPrepMethod_Other', 'NursDetFactor_CalendarDate','NursDetFactor_PreMonsoonShowers',
          'NursDetFactor_IrrigWaterAvailability','NursDetFactor_LabourAvailability', 'NursDetFactor_SeedAvailability',
          'TransDetFactor_LabourAvailability', 'TransDetFactor_CalendarDate','TransDetFactor_RainArrival',
          'TransDetFactor_IrrigWaterAvailability','TransDetFactor_SeedlingAge', 'CropbasalFerts_Urea','CropbasalFerts_DAP',
          'CropbasalFerts_Other', 'CropbasalFerts_NPK','CropbasalFerts_MoP', 'CropbasalFerts_NPKS', 'CropbasalFerts_SSP',
          'CropbasalFerts_None', 'FirstTopDressFert_Urea','FirstTopDressFert_DAP', 'FirstTopDressFert_NPK','FirstTopDressFert_NPKS',
          'FirstTopDressFert_SSP','FirstTopDressFert_Other', 'OrgFertilizers_Ganaura','OrgFertilizers_FYM',
          'OrgFertilizers_VermiCompost','OrgFertilizers_Pranamrit', 'OrgFertilizers_Ghanajeevamrit','OrgFertilizers_Jeevamrit',
          'OrgFertilizers_PoultryManure']

  for col in cols:
    df[col] = df[col].fillna(False)

  return df


def replace_numeric_nan(df):
  # Flag missing value for "Harv_hand_rent"
  df['Harv_hand_rent_NaN'] = df['Harv_hand_rent'].isna()
  # Flag instances where fertiliser was marked as used but the amount was zero
  df['BasalUrea_Inconsistency'] = df['CropbasalFerts_Urea'] & df['BasalUrea'].isna()
  # Replace NaN with 0 for columns where it makes sense
  fillna0 = ["2tdUrea","1tdUrea","Harv_hand_rent","Ganaura","CropOrgFYM","BasalDAP","BasalUrea"]

  for col in fillna0:
    # Fill missing values
    df[col] = df[col].fillna(0)

  return df


def count_na(df):
  # Create a new variable counting the number of missing values for each row (excluding outcome variables)
  df["Nb_of_NaN"] = df.drop(columns=["Yield"]).isnull().sum(axis=1)

  return df


def impute_missing(df):
  # Create imputation columns
  df["TpIrrigationSource_Imputed"] = df["TransplantingIrrigationSource"]
  df["TpIrrigationPowerSource_Imputed"] = df["TransplantingIrrigationPowerSource"]
  df["TpIrrigationHours_Imputed"] = df["TransplantingIrrigationHours"]
  df["TpIrrigationCost_Imputed"] = df["TransIrriCost"]
  df["SeedlingsPerPit_Imputed"] = df["SeedlingsPerPit"]
  df["1appDaysUrea_Imputed"] = df["1appDaysUrea"]
  df["2appDaysUrea_Imputed"] = df["2appDaysUrea"]

  # Impute missing categorical values with no statistical significance using mode
  df.loc[df["TpIrrigationSource_Imputed"].isnull()==True, "TpIrrigationSource_Imputed"] = df["TpIrrigationSource_Imputed"].mode().iloc[0]
  df.loc[df["TpIrrigationPowerSource_Imputed"].isnull()==True, "TpIrrigationPowerSource_Imputed"] = df["TpIrrigationPowerSource_Imputed"].mode().iloc[0]
  
  # Impute missing integers values with no statistical significance using median
  df.loc[df["TpIrrigationHours_Imputed"].isnull()==True, "TpIrrigationHours_Imputed"] = df["TpIrrigationHours_Imputed"].median()
  df.loc[df["TpIrrigationCost_Imputed"].isnull()==True, "TpIrrigationCost_Imputed"] = df["TpIrrigationCost_Imputed"].median()
  df.loc[df["SeedlingsPerPit_Imputed"].isnull()==True, "SeedlingsPerPit_Imputed"] = df["SeedlingsPerPit_Imputed"].median()
  df.loc[df["1appDaysUrea_Imputed"].isnull()==True, "1appDaysUrea_Imputed"] = df["1appDaysUrea_Imputed"].median()
  df.loc[df["2appDaysUrea_Imputed"].isnull()==True, "2appDaysUrea_Imputed"] = df["2appDaysUrea_Imputed"].median()

  return df


def handle_outliers(df):
  # Replace extreme outlier in SeedlingsPerPit with the next highest value
  df.loc[df["SeedlingsPerPit"]>22, "SeedlingsPerPit"] = 22
  # Replace extreme outlier in TransplantingIrrigationHours with reasonable value
  df.loc[df["TransplantingIrrigationHours"]>450, "TransplantingIrrigationHours"] = 450
  # Replace two extreme outliers in TransIrriCost with reasonable value
  df.loc[df["TransIrriCost"]>3000, "TransIrriCost"] = 3000
  # Replace extreme outlier in Harv_hand_rent with reasonable value
  df.loc[df["Harv_hand_rent"]>20000, "Harv_hand_rent"] = 20000
  # Replace extreme outlier in 1appDaysUrea with the next highest value
  df["1appDaysUrea"] = df["1appDaysUrea"].replace(332, 75)
  # Cap Ganaura at reasonable value and keep original column
  df["Ganaura_capped"] = df["Ganaura"]
  df.loc[df["Ganaura"]>50, "Ganaura_capped"] = 50

  return df


def rescale_entry_errors(df):
  # Split into training and test sets
  df_train = df[df['Yield'].isna() == False]
  df_test = df[df['Yield'].isna() == True]

  # Perform linear regression
  slope, intercept = np.polyfit(df_train['Acre'], df_train['Yield'], 1)
  # Calculate residuals
  predicted_y = slope * df_train['Acre'] + intercept
  residuals = df_train['Yield'] - predicted_y
  df_train['Residuals'] = residuals
  # Transform potential entry errors
  df_train['New_Yield'] = np.where(df_train['Residuals'] >= 1200, df_train['Yield'] / 10, df_train['Yield'])

  # Combine data
  df_test['New_Yield'] = np.nan
  df = pd.concat([df_train, df_test], axis=0)

  return df


def clean_data(train_path, test_path):
  df = load_data(train_path, test_path)
  df = fix_duplicate(df)
  df = adjust_datetime_columns(df)
  df = fix_errors(df)
  df = parse_categorical(df)
  df = replace_binary_nan(df)
  df = replace_numeric_nan(df)
  df = count_na(df)
  df = impute_missing(df)
  df = handle_outliers(df)
  df = rescale_entry_errors(df)

  return df