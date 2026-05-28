import pandas as pd

path = r"C:\Users\Parth Chauhan\Desktop\RAG_Project\PA_Business_rules_pc.xlsx"
df_all = pd.read_excel(path, sheet_name=None)

print("Sheets:", list(df_all.keys()))

for sheet_name, df in df_all.items():
    print(f"\n{'='*60}")
    print(f"SHEET: {sheet_name}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"{'='*60}")
    safe_name = sheet_name.replace(" ", "_").replace("/", "_")
    out_path = rf"C:\Users\Parth Chauhan\Desktop\RAG_Project\sheet_{safe_name}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(df.to_string())
    print(f"Written to: {out_path}")
    print(df.head(20).to_string())
