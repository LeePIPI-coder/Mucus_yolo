import pandas as pd
from pathlib import Path

data_dir = Path("/workspace/Get_3D_Class_Data")

for csv_file in sorted(data_dir.glob("Prediction_TP_FP_fold_*.csv")):
    df = pd.read_csv(csv_file)
    df_filtered = df[df["prediction_type"] != "TP"]
    df_filtered.to_csv(csv_file, index=False)
    print(f"{csv_file.name}: {len(df)} -> {len(df_filtered)} rows (removed {len(df) - len(df_filtered)} TP rows)")
