from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
path = BASE_DIR / "PA_Business_rules_pc.xlsx"
df_all = pd.read_excel(path, sheet_name=None)

print("Sheets:", list(df_all.keys()))

for sheet_name, df in df_all.items():
    safe_name = sheet_name.replace(" ", "_").replace("/", "_")
    out_path = BASE_DIR / f"sheet_{safe_name}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(df.to_string())
    print(f"Written to: {out_path}")

