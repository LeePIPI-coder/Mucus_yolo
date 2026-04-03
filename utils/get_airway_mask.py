from utils.SQL.SqlcipherStorageReader import SqlcipherStorageReader
from datetime import datetime
from utils.logging import get_logger
import shutil
import os
import tqdm

if __name__ == "__main__":
    input_path = "S:\AVIEWDB\mucus_plug"
    
    sql = SqlcipherStorageReader(input_path)
    all_patient = sql.get_all_series_list()
        
    # all_patient = [patient for patient in all_patient_list if datetime.strptime(patient.updated_at, "%Y-%m-%d %H:%M:%S") >= datetime.strptime(start_time, "%Y-%m-%d")]
    logger = get_logger("logs/get_airway_mask")
    for patient in tqdm.tqdm(all_patient, desc="提取气道掩码中"):
        check_time = datetime.strptime(patient.study_date_time, "%Y-%m-%d %H:%M:%S")
        patient_id = patient.patient_id
        time = f"{check_time.year}{check_time.month:02d}{check_time.day:02d}"   
        patient_path = patient.abs_path
        airway_mask_path = os.path.join(patient_path, "stor\objects\AirwaySeg.Obj.Airway.nii.gz")
        if not os.path.exists(airway_mask_path):
            logger.warning("该病例的气道mask不存在")
            continue
        new_airway_dir = f"S:/LJR/Mucus_airway_mask/{patient_id}_{time}"
        new_airway_path = os.path.join(new_airway_dir, f"{patient_id}_{time}_airway.nii.gz")
        if not os.path.exists(new_airway_dir):
            os.makedirs(new_airway_dir)
        shutil.copy(airway_mask_path, new_airway_path)