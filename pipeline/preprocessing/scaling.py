import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import warnings
warnings.filterwarnings("ignore")

def encode_cat_no_drop(df):
	# Apply one-hot encoding with the first column kept to categorical variables with missing entries
	columns_to_encode = ['Block','TransplantingIrrigationSource','TpIrrigationSource_Imputed','TpIrrigationPowerSource_Imputed',
	 					 'TransplantingIrrigationPowerSource','PCropSolidOrgFertAppMethod','MineralFertAppMethod_1',
	 					 'MineralFertAppMethod_2']
	df = pd.get_dummies(df, columns=columns_to_encode, drop_first=False)

	return df


def encode_cat_drop(df):
	# Apply one-hot encoding with the first column dropped to exclusive categorical variables with no missing entries
	columns_to_encode = ['District','CropEstMethod','Harv_method','Threshing_method','Stubble_use',
						 'LandPrepMethod_TractorPlough','LandPrepMethod_FourWheelTracRotavator','ThreshingSeason',
						 'LandPrepMethod_WetTillagePuddling','LandPrepMethod_BullockPlough','NursDetFactor_CalendarDate',
						 'NursDetFactor_PreMonsoonShowers','NursDetFactor_IrrigWaterAvailability',
						 'NursDetFactor_LabourAvailability','NursDetFactor_SeedAvailability','TransDetFactor_LabourAvailability',
						 'TransDetFactor_CalendarDate','TransDetFactor_RainArrival','TransDetFactor_IrrigWaterAvailability',
						 'TransDetFactor_SeedlingAge','CropbasalFerts_Urea','CropbasalFerts_DAP','CropbasalFerts_NPK','CropbasalFerts_NPKS',
						 'CropbasalFerts_SSP','CropbasalFerts_Other','CropbasalFerts_None','FirstTopDressFert_Urea',
						 'FirstTopDressFert_DAP','FirstTopDressFert_NPK','FirstTopDressFert_None','OrgFertilizers_Ganaura',
						 'OrgFertilizers_FYM','OrgFertilizers_VermiCompost','OrgFertilizers_Ghanajeevamrit','OrgFertilizers_None',
						 'RcNursEstDate_NaN','SeedlingsPerPit_NaN','TransplantingIrrigationSource_NaN',
						 'TransplantingIrrigationPowerSource_NaN','TransplantingIrrigationHours_NaN','PCropSolidOrgFertAppMethod_NaN',
						 'CropbasalFerts_NaN','FirstTopDressFert_NaN','TransIrriCost_NaN','StandingWater_NaN','OrgFertilizers_NaN',
						 'Ganaura_NaN','2appDaysUrea_NaN','MineralFertAppMethod_2_NaN','CropTillageMonth','1appDaysUrea_NaN',
						 'NursingMonth','SowTransplantMonth','HarvestMonth','ThreshingMonth']
	df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)

	return df


def scale_minmax(df):
	# Apply min-max scaling to discrete variables
	scaler = MinMaxScaler()
	columns_to_scale = ['CropTillageDepth','SeedlingsPerPit','SeedlingsPerPit_Imputed','StandingWater','NoFertilizerAppln',
					    '1appDaysUrea','Residue_perc','Nb_of_NaN','Num_LandPrepMethod','Num_NursDetFactor','Num_TransDetFactor',
					    'Num_OrgFertilizers','Num_CropbasalFerts','Num_TopDressFert','Latitude','Longitude','Elevation']
	df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

	return df


def scale_standard(df):
	# Apply standard scaling to approximately continuous variables
	scaler = StandardScaler()
	# Create new column for scaled acre
	df['Acre_Scaled'] = df['Acre']
	columns_to_scale = ['Acre_Scaled', 'CultLand','CropCultLand','TransplantingIrrigationHours','TransIrriCost','TransIrriCost_per_Acre',
						'Ganaura','CropOrgFYM','BasalDAP','BasalUrea','1tdUrea','2tdUrea','2appDaysUrea','Harv_hand_rent','Residue_length',
					    'Ganaura_capped','TransplantingIrrigationHours_per_Acre','TpIrrigationHours_Imputed_per_Acre','TpIrrigationCost_Imputed',
					    'Ganaura_per_Acre','CropOrgFYM_per_Acre','BasalDAP_per_Acre','BasalUrea_per_Acre','1tdUrea_per_Acre','2tdUrea_per_Acre',
					    'Harv_hand_rent_per_Acre','Days_bw_Nurs_SowTransp','Days_bw_Nurs_Harv','Days_bw_Nurs_Till','Days_bw_Till_SowTransp',
					    'Days_bw_Till_Harv','Days_bw_SowTransp_Harv','Days_bw_Harv_Thresh','NursingDate_ModeDiff','TillageDate_ModeDiff',
					    'SowTransplantDate_ModeDiff','HarvestDate_ModeDiff','ThreshingDate_ModeDiff','Days_bw_Nurs_SowTransp_ModeDiff',
					    'Days_bw_Nurs_Harv_ModeDiff','Days_bw_Nurs_Till_ModeDiff','Days_bw_Till_SowTransp_ModeDiff','Days_bw_Till_Harv_ModeDiff',
					    'Days_bw_SowTransp_Harv_ModeDiff','Days_bw_Harv_Thresh_ModeDiff','Total_Crop_Cycle_Duration','2appDaysUrea_MeanDiff',
					    '2appDaysUrea_Imputed_MeanDiff','1appDaysUrea_Imputed','2appDaysUrea_Imputed','TpIrrigationHours_Imputed',
					    'TpIrrigationCost_Imputed_per_Acre','TransIrriCost_per_Acre_capped','TpIrrigationCost_Imputed_per_Acre_capped',
					    'Days_bw_Nurs_SowTransp_Imputed','Days_bw_Nurs_Harv_Imputed','Days_bw_Nurs_Till_Imputed','NursingDate_ModeDiff_Imputed',
					    'Days_bw_Nurs_SowTransp_ModeDiff_Imputed','Days_bw_Nurs_Harv_ModeDiff_Imputed','Days_bw_Nurs_Till_ModeDiff_Imputed',
					    'TransplantingIrrigationHours_per_Acre_capped','TpIrrigationHours_Imputed_per_Acre_capped','Ganaura_per_Acre_capped',
					    'CropOrgFYM_per_Acre_capped','BasalDAP_per_Acre_capped','BasalUrea_per_Acre_capped','1tdUrea_per_Acre_capped',
					    '2tdUrea_per_Acre_capped','Harv_hand_rent_per_Acre_capped']
	df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

	return df


def scale_features(df):
    df = encode_cat_drop(df)
    df = encode_cat_no_drop(df)
    df = scale_minmax(df)
    df = scale_standard(df)

    return df