import pandas as pd
import numpy as np
import calendar
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")


def get_scaled_per_acre(df):
    per_acre_cols = ["TransplantingIrrigationHours", "TpIrrigationHours_Imputed", "TransIrriCost", "Ganaura", "CropOrgFYM",
                   "BasalDAP", "BasalUrea", "1tdUrea", "2tdUrea", "Harv_hand_rent", "Yield", "New_Yield", "TransIrriCost",
                   "TpIrrigationCost_Imputed"]
    for col in per_acre_cols:
        label = str(col) + "_per_Acre"
        df[label] = df[col] / df["Acre"]

    return df


def handle_per_acre_outliers(df):
    # Select columns to be capped
    cols = ["TransplantingIrrigationHours_per_Acre","TpIrrigationHours_Imputed_per_Acre","TpIrrigationCost_Imputed_per_Acre",
            "TransIrriCost_per_Acre","Ganaura_per_Acre","CropOrgFYM_per_Acre","BasalDAP_per_Acre","BasalUrea_per_Acre",
            "1tdUrea_per_Acre","2tdUrea_per_Acre","Harv_hand_rent_per_Acre"]

    for col in cols:
        # Replace extreme outliers with the next highest value and keep original column
        label = col + "_capped"
        df[label] = df[col]
        cap_value = np.percentile(df[col].dropna(), 95)
        df.loc[df[col]>cap_value, label] = cap_value

        # Create extreme value flags
        label = col + "_XValue"
        df[label] = df[col + "_capped"] != df[col]

    return df


def get_nan_indicators(df):
    nan_cols = ['RcNursEstDate','SeedlingsPerPit','TransplantingIrrigationSource','TransplantingIrrigationPowerSource',
                'TransplantingIrrigationHours','PCropSolidOrgFertAppMethod','CropbasalFerts','FirstTopDressFert', 
                'TransIrriCost','StandingWater','OrgFertilizers','Ganaura','2appDaysUrea','MineralFertAppMethod_2',
                '1appDaysUrea','TransIrriCost']
    for col in nan_cols:
        label = f"{col}_NaN"
        df[label] = df[col].isna()

    return df


def get_date_step_distance(df):
    # Calculate number of days between key stages of the crop cycle
    df["Days_bw_Nurs_SowTransp"] = df["SeedingSowingTransplanting"] - df["RcNursEstDate"]
    df["Days_bw_Nurs_Harv"] = df["Harv_date"] - df["RcNursEstDate"]
    df["Days_bw_Nurs_Till"] = df["CropTillageDate"] - df["RcNursEstDate"]
    df["Days_bw_Till_SowTransp"] = df["SeedingSowingTransplanting"] - df["CropTillageDate"]
    df["Days_bw_Till_Harv"] = df["Harv_date"] - df["CropTillageDate"]
    df["Days_bw_SowTransp_Harv"] = df["Harv_date"] - df["SeedingSowingTransplanting"]
    df["Days_bw_Harv_Thresh"] = df["Threshing_date"] - df["Harv_date"]

    # Re-format new variables
    days_cols = ["Days_bw_Nurs_SowTransp","Days_bw_Nurs_Harv","Days_bw_Nurs_Till","Days_bw_Till_SowTransp",
               "Days_bw_Till_Harv","Days_bw_SowTransp_Harv","Days_bw_Harv_Thresh"]
    for col in days_cols:
        df[col] = df[col].astype(str).str[:-5]
        df.loc[df[col]=='', col] = np.nan # Format missing values

    # Format date distances as floats
    for col in days_cols:
        df[col] = df[col].astype(float)

    return(df)


def calculate_date_difference(ref, col, month_day):
    # Calculate the difference between the dates in a column and a specific date ignoring the year
    col_year = ref.dt.year
    ref_date = pd.to_datetime(col_year.astype(str) + '-' + month_day, format="%Y-%m-%d")

    result = (col - ref_date).dt.days

    return result


def get_date_mode_difference(df):
    ref = df['CropTillageDate']
    # Calculate the difference from the annual mode date for each step
    df["NursingDate_ModeDiff"] = calculate_date_difference(
        ref, df["RcNursEstDate"], str(df['RcNursEstDate'].mode()[0])[5:10])
    df["TillageDate_ModeDiff"] = calculate_date_difference(
        ref, df["CropTillageDate"], str(df['CropTillageDate'].mode()[0])[5:10])
    df["SowTransplantDate_ModeDiff"] = calculate_date_difference(
        ref, df["SeedingSowingTransplanting"], str(df['SeedingSowingTransplanting'].mode()[0])[5:10])
    df["HarvestDate_ModeDiff"] = calculate_date_difference(
        ref, df["Harv_date"], str(df['Harv_date'].mode()[0])[5:10])
    df["ThreshingDate_ModeDiff"] = calculate_date_difference(
        ref, df["Threshing_date"], str(df['Threshing_date'].mode()[0])[5:10])

    return df


def get_days_mode_difference(df):
    # Calculate the difference from the mode days for each variable
    df["Days_bw_Nurs_SowTransp_ModeDiff"] = df["Days_bw_Nurs_SowTransp"] - df["Days_bw_Nurs_SowTransp"].mode()[0]
    df["Days_bw_Nurs_Harv_ModeDiff"] = df["Days_bw_Nurs_Harv"] - df["Days_bw_Nurs_Harv"].mode()[0]
    df["Days_bw_Nurs_Till_ModeDiff"] = df["Days_bw_Nurs_Till"] - df["Days_bw_Nurs_Till"].mode()[0]
    df["Days_bw_Till_SowTransp_ModeDiff"] = df["Days_bw_Till_SowTransp"] - df["Days_bw_Till_SowTransp"].mode()[0]
    df["Days_bw_Till_Harv_ModeDiff"] = df["Days_bw_Till_Harv"] - df["Days_bw_Till_Harv"].mode()[0]
    df["Days_bw_SowTransp_Harv_ModeDiff"] = df["Days_bw_SowTransp_Harv"] - df["Days_bw_SowTransp_Harv"].mode()[0]
    df["Days_bw_Harv_Thresh_ModeDiff"] = df["Days_bw_Harv_Thresh"] - df["Days_bw_Harv_Thresh"].mode()[0]

    return df


def get_mean_difference(df):
    # Calculate the difference from the mean for each variable
    df["2appDaysUrea_MeanDiff"] = df["2appDaysUrea"] - df["2appDaysUrea"].mean()
    df["2appDaysUrea_Imputed_MeanDiff"] = df["2appDaysUrea_Imputed"] - df["2appDaysUrea_Imputed"].mean()

    return df
 

def get_months(df):
    # Extract months from date variables
    df["CropTillageMonth"] = df["CropTillageDate"].dt.month_name()
    df["NursingMonth"] = df["RcNursEstDate"].dt.month_name()
    df["SowTransplantMonth"] = df["SeedingSowingTransplanting"].dt.month_name()
    df["HarvestMonth"] = df["Harv_date"].dt.month_name()
    df["ThreshingMonth"] = df["Threshing_date"].dt.month_name()

    # Fix low value months
    df["CropTillageMonth"] = np.where(df['CropTillageMonth'] == 'May', 'June', df["CropTillageMonth"])
    df["HarvestMonth"] = np.where(df['HarvestMonth'] == 'September', 'October', df['HarvestMonth'])
    df["HarvestMonth"] = np.where(df['HarvestMonth'].isin(['January', 'February', 'March']), 'December', df['HarvestMonth'])

    return df


def impute_nursing(df):
    # Create imputation columns
    columns_to_impute = ["Days_bw_Nurs_SowTransp", "Days_bw_Nurs_Harv", "Days_bw_Nurs_Till", "NursingDate_ModeDiff",
                         "Days_bw_Nurs_SowTransp_ModeDiff", "Days_bw_Nurs_Harv_ModeDiff", "Days_bw_Nurs_Till_ModeDiff"]

    # Create imputation columns and impute missing integer values
    for col in columns_to_impute:
        df[f"{col}_Imputed"] = df[col]
        df.loc[df[f"{col}_Imputed"].isnull(), f"{col}_Imputed"] = df[f"{col}_Imputed"].median()

    return df


def assign_season(col):
    # Assign seasons based Bihars's seasonal patterns
    conditions = [
        col.isin(['June','July','August','September']),
        col.isin(['October','November','December']),
        col.isin(['January','February','March']),
        col.isin(['April','May'])
    ]
    choices = ['Monsoon','Post-Monsoon','Winter','Summer']
    result = np.select(conditions, choices, default=np.nan)
    return result


def get_seasons(df):
    month_cols = ['CropTillage','Nursing','SowTransplant','Harvest','Threshing']

    for col in month_cols:
        df[f'{col}Season'] = assign_season(df[f'{col}Month'])
        df[f'{col}Season'].replace('nan', np.nan, inplace=True)

    return df


def get_total_crop_cycle_duration(df):
    df["Total_Crop_Cycle_Duration"] = df["Days_bw_Till_SowTransp"] + df["Days_bw_SowTransp_Harv"] + df["Days_bw_Harv_Thresh"]

    return df

def get_num_options_used(df):
    landprep = ["LandPrepMethod_TractorPlough","LandPrepMethod_FourWheelTracRotavator",
                "LandPrepMethod_WetTillagePuddling","LandPrepMethod_BullockPlough","LandPrepMethod_Other"]
    df["Num_LandPrepMethod"] = df[landprep].sum(axis=1)

    nurserydet = ["NursDetFactor_CalendarDate","NursDetFactor_PreMonsoonShowers","NursDetFactor_IrrigWaterAvailability",
                  "NursDetFactor_LabourAvailability","NursDetFactor_SeedAvailability"]
    df["Num_NursDetFactor"] = df[landprep].sum(axis=1)

    transdet = ["TransDetFactor_LabourAvailability","TransDetFactor_CalendarDate","TransDetFactor_RainArrival",
                "TransDetFactor_IrrigWaterAvailability","TransDetFactor_SeedlingAge"]
    df["Num_TransDetFactor"] = df[transdet].sum(axis=1)

    orgfertilizers = ["OrgFertilizers_Ganaura","OrgFertilizers_FYM","OrgFertilizers_VermiCompost",
                      "OrgFertilizers_Pranamrit","OrgFertilizers_Ghanajeevamrit","OrgFertilizers_Jeevamrit"]
    df["Num_OrgFertilizers"] = df[orgfertilizers].sum(axis=1)

    fertilizer_types = ["CropbasalFerts_Urea","CropbasalFerts_DAP","CropbasalFerts_NPK","CropbasalFerts_MoP",
                        "CropbasalFerts_NPKS","CropbasalFerts_SSP","CropbasalFerts_Other"]
    df["Num_CropbasalFerts"] = df[fertilizer_types].sum(axis=1)

    fertilizer_types2 = ["FirstTopDressFert_Urea","FirstTopDressFert_DAP","FirstTopDressFert_NPK",
                         "FirstTopDressFert_NPKS","FirstTopDressFert_SSP","FirstTopDressFert_Other"]
    df["Num_TopDressFert"] = df[fertilizer_types2].sum(axis=1)

    return df


def get_geography(df):
    # Get dictionary of lattitude, longitude, and elevation
    latlon_dict = {"Chehrakala": [25.885679,85.380560,55],"Garoul": [28.5002047255,77.2918956199, 55],"Gaya": [24.795452,84.999431,116.207],
                   "Gurua": [24.6696,84.7720,114],"Jamui": [24.919515,86.224718,89.048],"Khaira": [24.8715,86.2071,79],
                   "Lohra": [24.9370,86.1764,78], "Mahua": [25.8147,85.3969,54], "Noorsarai": [25.2748,85.4569,63],
                   "Rajgir": [25.0262,85.4174,67], "Wazirganj": [24.8029,85.2436,112]}

    # Create a mapping from "Block" to the corresponding lattitude, longitude, and elevation
    block_map = {k: v for k, v in latlon_dict.items() if k in df['Block'].unique()}

    # Create new columns for latitude, longitude, and elevation
    df['Latitude'] = df['Block'].map(lambda x: block_map[x][0] if x in block_map else np.nan)
    df['Longitude'] = df['Block'].map(lambda x: block_map[x][1] if x in block_map else np.nan)
    df['Elevation'] = df['Block'].map(lambda x: block_map[x][2] if x in block_map else np.nan)

    return df


def get_features(df):
    df = get_scaled_per_acre(df)
    df = handle_per_acre_outliers(df)
    df = get_nan_indicators(df)
    df = get_date_step_distance(df)
    df = get_date_mode_difference(df)
    df = get_days_mode_difference(df)
    df = get_mean_difference(df)
    df = get_months(df)
    df = impute_nursing(df)
    df = get_seasons(df)
    df = get_total_crop_cycle_duration(df)
    df = get_num_options_used(df)
    df = get_geography(df)

    return df