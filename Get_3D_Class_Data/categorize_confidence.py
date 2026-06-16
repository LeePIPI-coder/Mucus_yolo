import pandas as pd
from pathlib import Path

data_dir = Path("/workspace/Get_3D_Class_Data/一阶段预测数据FROC结果")

for csv_file in sorted(data_dir.glob("Prediction_TP_FP_fold_*.csv")):
    df = pd.read_csv(csv_file)

    col = "detector_score"

    def classify(score):
        if score > 0.8:
            return "高置信度"
        elif score >= 0.5:
            return "中置信度"
        else:
            return "低置信度"

    # 先初始化
    df["confidence_level"] = None

    # 仅对 FP 行做 classify
    fp_mask = df["prediction_type"] == "FP"

    df.loc[fp_mask, "confidence_level"] = (
        df.loc[fp_mask, col].apply(classify)
    )

    df.to_csv(csv_file, index=False)

    n_high = (df["confidence_level"] == "高置信度").sum()
    n_mid = (df["confidence_level"] == "中置信度").sum()
    n_low = (df["confidence_level"] == "低置信度").sum()
    print(f"{csv_file.name}: {len(df)} rows -> High={n_high} | Mid={n_mid} | Low={n_low}")
