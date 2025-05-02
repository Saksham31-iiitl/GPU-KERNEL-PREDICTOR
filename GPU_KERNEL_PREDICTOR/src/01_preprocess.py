

import os
import pandas as pd

def main():
   
    ROOT_DIR  = os.path.dirname(os.path.dirname(__file__))
    RAW_CSV   = os.path.join(ROOT_DIR, "data", "kernels_dataset.csv")
    PROC_DIR  = os.path.join(ROOT_DIR, "data", "processed")
    CLEAN_CSV = os.path.join(PROC_DIR, "kernels_clean.csv")

    
    print(f"Loading raw data from {RAW_CSV}...")
    df = pd.read_csv(RAW_CSV)
    print(f"▶ Raw shape: {df.shape}")
    print(f"▶ Raw columns: {df.columns.tolist()}")

   
    df = df.dropna().reset_index(drop=True)
    print(f"▶ After dropna: {df.shape}")

   
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(" ", "_", regex=False)
          .str.replace(r"[^\w_]", "", regex=True)
    )
    print(f"▶ Clean columns: {df.columns.tolist()}")

  
    os.makedirs(PROC_DIR, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)
    print(f"✅ Written cleaned data to: {CLEAN_CSV}")

if __name__ == "__main__":
    main()
