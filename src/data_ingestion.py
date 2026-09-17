import os
import glob
import pandas as pd
import numpy as np

def extract_financials(file_path):
    symbol = os.path.basename(file_path).replace('.xlsx', '')
    
    try:
        df = pd.read_excel(file_path, sheet_name='Data Sheet')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    # Extract Metadata
    industry = "Unknown"
    for idx, row in df.iterrows():
        if str(row.iloc[0]).strip() == "Industry":
            industry = row.iloc[1]
            break

    # We will look for PROFIT & LOSS, BALANCE SHEET, CASH FLOW
    sections = ["PROFIT & LOSS", "BALANCE SHEET", "CASH FLOW"]
    
    data_records = {} # key: date, value: dict of metrics
    
    current_section = None
    dates = []
    
    for idx, row in df.iterrows():
        metric = str(row.iloc[0]).strip()
        
        if metric in sections:
            current_section = metric
            continue
            
        if current_section and metric == "Report Date":
            # Extract dates from this row
            dates = []
            for col_idx in range(1, len(row)):
                val = row.iloc[col_idx]
                if pd.notna(val) and (isinstance(val, pd.Timestamp) or str(val).startswith('20')):
                    # Screener sometimes has 'TTM' as a date. Let's keep it or ignore it.
                    # We will parse it to year if possible, or just string.
                    if isinstance(val, pd.Timestamp):
                        dates.append((col_idx, val.year))
                    else:
                        dates.append((col_idx, str(val)[:4]))
            continue
            
        if current_section and pd.notna(metric) and metric != "nan":
            # Extract values for the known dates
            for col_idx, date_label in dates:
                val = row.iloc[col_idx]
                if pd.notna(val):
                    key = (symbol, date_label)
                    if key not in data_records:
                        data_records[key] = {"Company": symbol, "Year": date_label, "Industry": industry}
                    data_records[key][metric] = val
                    
    # Convert to list of dicts
    return list(data_records.values())

def main():
    files = glob.glob('screener_exports/*.xlsx')
    all_data = []
    
    print(f"Processing {len(files)} files...")
    
    for i, f in enumerate(files):
        records = extract_financials(f)
        all_data.extend(records)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(files)}")
            
    final_df = pd.DataFrame(all_data)
    
    # Save to data directory
    output_path = 'data/raw_panel_data.csv'
    final_df.to_csv(output_path, index=False)
    print(f"\nData ingestion complete. Saved shape: {final_df.shape} to {output_path}")

if __name__ == "__main__":
    main()
