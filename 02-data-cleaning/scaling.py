import pandas as pd
import numpy as np
import calendar
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import warnings
warnings.filterwarnings("ignore")

def encode_cat_drop(df):
	# Apply one-hot encoding with the first column dropped to categorical variables
	columns_to_encode = ['District','Block','CropEstMethod','Harv_method','Threshing_method','Stubble_use',
						 'PCropSolidOrgFertAppMethod','MineralFertAppMethod']
	df = pd.get_dummies(df, columns=columns_to_encode)

	return df


def encode_cat_no_drop(df):
	# Apply one-hot encoding with the first column kept to categorical variables with missing entries
	columns_to_encode = ['TransplantingIrrigationSource','TransplantingIrrigationPowerSource',
						 'LandPrepMethod_TractorPlough','LandPrepMethod_FourWheelTracRotavator',
						 'LandPrepMethod_WetTillagePuddling','LandPrepMethod_BullockPlough','NursDetFactor_CalendarDate',
						 'NursDetFactor_PreMonsoonShowers','NursDetFactor_IrrigWaterAvailability',
						 'NursDetFactor_LabourAvailability','NursDetFactor_SeedAvailability','TransDetFactor_LabourAvailability',
						 'TransDetFactor_CalendarDate','TransDetFactor_RainArrival','TransDetFactor_IrrigWaterAvailability',
						 'TransDetFactor_SeedlingAge','TransDetFactor_CalendarDate','TransDetFactor_RainArrival',
						 'CropbasalFerts_Urea','CropbasalFerts_DAP','CropbasalFerts_NPK','CropbasalFerts_NPKS',
						 'CropbasalFerts_SSP','CropbasalFerts_Other','CropbasalFerts_None','FirstTopDressFert_Urea',
						 'FirstTopDressFert_DAP','FirstTopDressFert_NPK','FirstTopDressFert_None','OrgFertilizers_Ganaura',
						 'OrgFertilizers_FYM','OrgFertilizers_VermiCompost','OrgFertilizers_Ghanajeevamrit','OrgFertilizers_None',
						 'TpIrrigationSource_Imputed','TpIrrigationPowerSource_Imputed','RcNursEstDate_NaN','SeedlingsPerPit_NaN',
						 'TransplantingIrrigationSource_NaN','TransplantingIrrigationPowerSource_NaN','TransplantingIrrigationHours_NaN',
						 'CropTillageMonth','NursingMonth','SowTransplantMonth','HarvestMonth','ThreshingMonth']
	df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)

	return df


def scale_minmax(df):
	# Apply min-max scaling to discrete variables
	scaler = MinMaxScaler()
	columns_to_scale= ['CropTillageDepth','SeedlingsPerPit','StandingWater','NoFertilizerAppln','1appDaysUrea','Residue_perc',
					   'Nb_of_NaN','Num_LandPrepMethod','Num_NursDetFactor','Num_TransDetFactor','Num_OrgFertilizers',
					   'Num_CropbasalFerts','Num_TopDressFert','Latitude','Longitude','Elevation']
	df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

	return df


def scale_standard(df):
	# Apply standard scaling to approximately continuous variables
	scaler = StandardScaler()
	columns_to_scale= ['CultLand','CropCultLand','TransplantingIrrigationHours','TransIrriCost','Ganaura','CropOrgFYM','BasalDAP',
					   'BasalUrea','1tdUrea','2tdUrea','2appDaysUrea','Harv_hand_rent','Residue_length','Acre','TpIrrigationHours_Imputed',
					   'Ganaura_capped','TransplantingIrrigationHours_per_Acre','TpIrrigationHours_Imputed_per_Acre','TransIrriCost_per_Acre',
					   'Ganaura_per_Acre','CropOrgFYM_per_Acre','BasalDAP_per_Acre','BasalUrea_per_Acre','1tdUrea_per_Acre','2tdUrea_per_Acre',
					   'Harv_hand_rent_per_Acre','Days_bw_Nurs_SowTransp','Days_bw_Nurs_Harv','Days_bw_Nurs_Till','Days_bw_Till_SowTransp',
					   'Days_bw_Till_Harv','Days_bw_SowTransp_Harv','Days_bw_Harv_Thresh','NursingDate_ModeDiff','TillageDate_ModeDiff',
					   'SowTransplantDate_ModeDiff','HarvestDate_ModeDiff','ThreshingDate_ModeDiff','Days_bw_Nurs_SowTransp_ModeDiff',
					   'Days_bw_Nurs_Harv_ModeDiff','Days_bw_Nurs_Till_ModeDiff','Days_bw_Till_SowTransp_ModeDiff','Days_bw_Till_Harv_ModeDiff',
					   'Days_bw_SowTransp_Harv_ModeDiff','Days_bw_Harv_Thresh_ModeDiff','CropTillageMonth','Total_Crop_Cycle_Duration']
	df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

	return df

def drop_unusable_vars(df):
	df = df.drop(columns=['LandPreparationMethod','CropTillageDate','RcNursEstDate','SeedingSowingTransplanting','NursDetFactor'
						  'TransDetFactor','OrgFertilizers','CropbasalFerts','FirstTopDressFert','Harv_date','Threshing_date',
						  'LandPrepMethod_Other','CropbasalFerts_MoP','FirstTopDressFert_NPKS','FirstTopDressFert_SSP',
						  'FirstTopDressFert_Other','OrgFertilizers_Pranamrit','OrgFertilizers_Jeevamrit','OrgFertilizers_PoultryManure',
						  'CropTillageSeason','NursingSeason','SowTransplantSeason','HarvestSeason'])

	return df


def do_scaling(df):
    df = encode_cat_drop(df)
    df = encode_cat_no_drop(df)
    df = scale_minmax(df)
    df = scale_standard(df)
    df = drop_unusable_vars(df)

    return df