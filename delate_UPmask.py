from utils.SQL.SqlcipherStorageReader import SqlcipherStorageReader
from utils.logging import get_logger
from pathlib import Path
import os

logger = get_logger("logs/delate_UPmask")

sql_path= r"S:\AVIEWDB\mucus_plug"
sql = SqlcipherStorageReader(sql_path)
all_patient_list = sql.get_all_series_list()
for i, patient in enumerate(all_patient_list):
    logger.info(f"正在处理患者 {i+1}/{len(all_patient_list)}: {patient.patient_id}")
    logger.info(f"患者路径: {patient.abs_path}")
    patient_dir = Path(patient.abs_path)
    results_dir = patient_dir / 'stor/results'
    for filename in os.listdir(results_dir):
        if filename.startswith('lesionAnnot3D'):
            file_path = os.path.join(results_dir, filename)
            if Path(file_path).is_file():
                logger.info(f"删除文件: {file_path}")
                os.remove(file_path)
