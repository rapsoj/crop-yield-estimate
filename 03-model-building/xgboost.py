import pandas as pd
import numpy as np
import calendar
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

def reorder_cols(df):
  # Re-order columns for clarity
  df = df[['ID','Set','District','Block','CropTillageDate','CropTillageMonth','CropTillageDepth','CropEstMethod',
          'RcNursEstDate','NursingMonth', 'SeedingSowingTransplanting','SowTransplantMonth','SeedlingsPerPit',
          'TransplantingIrrigationHours','TransplantingIrrigationSource', 'TransplantingIrrigationPowerSource',
          'TransIrriCost','TransIrriCost_per_Acre','StandingWater','Ganaura','Ganaura_per_Acre','CropOrgFYM', 
          'CropOrgFYM_per_Acre','PCropSolidOrgFertAppMethod','NoFertilizerAppln','BasalDAP','BasalDAP_per_Acre',
          'BasalUrea','BasalUrea_per_Acre','MineralFertAppMethod','1tdUrea','1tdUrea_per_Acre','1appDaysUrea',
          '2tdUrea','2tdUrea_per_Acre','2appDaysUrea','MineralFertAppMethod.1','Harv_method','Harv_date',
          'HarvestMonth','Harv_hand_rent','Harv_hand_rent_per_Acre','Threshing_date','ThreshingMonth',
          'Threshing_method','Residue_length','Residue_perc','Stubble_use','Acre','Yield','Yield_per_Acre',
          # Parsed categoricals
          'LandPrepMethod_TractorPlough','LandPrepMethod_FourWheelTracRotavator','LandPrepMethod_WetTillagePuddling',
          'LandPrepMethod_BullockPlough','LandPrepMethod_Other','NursDetFactor_CalendarDate',
          'NursDetFactor_PreMonsoonShowers','NursDetFactor_IrrigWaterAvailability',
          'NursDetFactor_LabourAvailability','NursDetFactor_SeedAvailability','TransDetFactor_LabourAvailability',
          'TransDetFactor_CalendarDate','TransDetFactor_RainArrival','TransDetFactor_IrrigWaterAvailability',
          'TransDetFactor_SeedlingAge','CropbasalFerts_Urea','CropbasalFerts_DAP','CropbasalFerts_Other',
          'CropbasalFerts_NPK','CropbasalFerts_MoP','CropbasalFerts_NPKS','CropbasalFerts_SSP',
          'CropbasalFerts_None','FirstTopDressFert_Urea','FirstTopDressFert_DAP','FirstTopDressFert_NPK',
          'FirstTopDressFert_NPKS','FirstTopDressFert_SSP','FirstTopDressFert_Other','OrgFertilizers_Ganaura',
          'OrgFertilizers_FYM','OrgFertilizers_VermiCompost','OrgFertilizers_Pranamrit','OrgFertilizers_Ghanajeevamrit',
          'OrgFertilizers_Jeevamrit','OrgFertilizers_PoultryManure','Days_bw_Nurs_SowTransp',
          'Days_bw_SowTransp_Harv','Days_bw_Harv_Thresh',
          # Other additions
          'Nb_of_NaN']]

  return df