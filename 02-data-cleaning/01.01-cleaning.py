import pandas as pd
import numpy as np
import calendar
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")


def load_data(train_path, test_path):
  train = pd.read_csv(train_path)
  test = pd.read_csv(test_path)
    
  train["Set"] = "train"
  test["Set"] = "test"
    
  df = pd.concat([train, test])
    
  return df


def adjust_datetime_columns(df):
  datetime_cols = ["CropTillageDate", "RcNursEstDate", "SeedingSowingTransplanting", "Harv_date", "Threshing_date"]
  for col in datetime_cols:
    df[col] = pd.to_datetime(df[col])

  return df


def fix_errors(df):
  # Fix district typo
  df.loc[(df["District"]=="Jamui") & (df["Block"]=="Gurua")].index
  df.loc[2177,"District"] = "Gaya"

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


def handle_outliers(df):
  # Replace two extreme outliers in SeedlingsPerPit with the next highest value
  df.loc[df["SeedlingsPerPit"]>22, "SeedlingsPerPit"] = 22
  # Cap TransplantingIrrigationHours at reasonable value
  df.loc[df["TransplantingIrrigationHours"]> 50, "TransplantingIrrigationHours"] = 450
  # Cap TransIrriCost at reasonable value
  df.loc[df["TransIrriCost"]>3000, "TransIrriCost"] = 3000
  # Cap Ganaura at reasonable value and leave original column
  df.loc[df["Ganaura"]>50, "Ganaura_capped"] = 50
  # Replace extreme outlier in 1appDaysUrea with the next highest value
  df["1appDaysUrea"] = df["1appDaysUrea"].replace(332, 75)
  # Cap Harv_hand_rent at reasonable value
  df.loc[df["Harv_hand_rent"]>20000, "Harv_hand_rent"] = 20000

  return df


def scale_per_acre(df):
  per_acre_cols = ["TransIrriCost", "Ganaura", "CropOrgFYM", "BasalDAP", "BasalUrea",
                   "1tdUrea", "2tdUrea", "Harv_hand_rent", "Yield"]
  for col in per_acre_cols:
    label = str(col) + "_per_Acre"
    df[label] = df[col] / df["Acre"]

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
  df["NursDetFactor_LabourAvailability"] = df["NursDetFactor"].str.contains("LabourAvailability" or "LaborAvailability")
  df["NursDetFactor_SeedAvailability"] = df["NursDetFactor"].str.contains("SeedAvailability")

  df["TransDetFactor_LabourAvailability"] = df["TransDetFactor"].str.contains("LabourAvailability" or "LaborAvailability")
  df["TransDetFactor_CalendarDate"] = df["TransDetFactor"].str.contains("CalendarDate")
  df["TransDetFactor_RainArrival"] = df["TransDetFactor"].str.contains("RainArrival")
  df["TransDetFactor_IrrigWaterAvailability"] = df["TransDetFactor"].str.contains("IrrigWaterAvailability")
  df["TransDetFactor_SeedlingAge"] = df["TransDetFactor"].str.contains("SeedlingAge")

  df["CropbasalFerts"] = df["CropbasalFerts"].fillna("None")
  fertilizer_types = ["Urea","DAP","Other","NPK","MoP","NPKS","SSP","None"]
  for fertilizer in fertilizer_types:
    label = "CropbasalFerts_" + fertilizer
    df[label] = df["CropbasalFerts"].str.contains(fertilizer)

  df["FirstTopDressFert"] = df["FirstTopDressFert"].fillna("None")
  fertilizer_types2 = ["Urea","DAP","NPK","NPKS","SSP","Other"]
  for fertilizer in fertilizer_types2:
    label = "FirstTopDressFert_" + fertilizer
    df[label] = df["FirstTopDressFert"].str.contains(fertilizer)

  df["OrgFertilizers"] = df["OrgFertilizers"].fillna("None")
  orgfertilizers = ["Ganaura","FYM","VermiCompost","Pranamrit","Ghanajeevamrit","Jeevamrit","PoultryManure"]
  for fertilizer in orgfertilizers:
    label = "OrgFertilizers_" + fertilizer
    df[label] = df["OrgFertilizers"].str.contains(fertilizer)

  # Drop unparsed variables
  df = df.drop(columns=["LandPreparationMethod","NursDetFactor","TransDetFactor","CropbasalFerts","FirstTopDressFert","OrgFertilizers"])

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
  # Replace NaN with 0 for columns where it makes sense
  fillna0 = ["2tdUrea","1tdUrea","Harv_hand_rent","Ganaura","CropOrgFYM","BasalDAP","BasalUrea"]

  for col in fillna0:
    df[col] = df[col].fillna(0)

  return df


def count_na(df):
  # Create a new variable counting the number of missing values for each row (excluding outcome variables)
  df["Nb_of_NaN"] = df.drop(columns=["Yield","Yield_per_Acre"]).isnull().sum(axis=1)

  return df


def impute_missing(df):
  # Impute missing categorical values with no statistical significance using mode
  df.loc[df["TransplantingIrrigationSource"].isnull()==True, "TransplantingIrrigationSource"] = df["TransplantingIrrigationSource"].mode()
  df.loc[df["TransplantingIrrigationPowerSource"].isnull()==True, "TransplantingIrrigationPowerSource"] = df["TransplantingIrrigationPowerSource"].mode()
  # Impute missing integers values with no statistical significance using median
  df.loc[df["TransplantingIrrigationHours"].isnull()==True, "TransplantingIrrigationHours"] = df["TransplantingIrrigationHours"].median()

  return df


def clean_data(train_path, test_path):
  df = load_data(train_path, test_path)
  df = adjust_datetime_columns(df)
  df = fix_errors(df)
  df = handle_outliers(df)
  df = scale_per_acre(df)
  df = parse_categorical(df)
  df = replace_binary_nan(df)
  df = replace_numeric_nan(df)
  df = count_na(df)
  df = impute_missing(df)

  return df