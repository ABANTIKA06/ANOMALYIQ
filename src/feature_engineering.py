import pandas as pd
import numpy as np
import os

def main():
    input_path = 'data/raw_panel_data.csv'
    output_path = 'data/ml_features.csv'
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please run data_ingestion.py first.")
        return
        
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    
    # Identify available columns (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]
    
    # Sort by Company and Year for time-series calculations
    df.sort_values(by=['Company', 'Year'], inplace=True)
    
    # Initialize ml_features dataframe with index columns
    features = df[['Company', 'Year', 'Industry']].copy()
    
    # 1. Accruals Ratio = (Net profit - Cash from Operating Activity) / Total Assets
    # Let's find the exact column names
    net_profit_col = 'Net profit'
    cfo_col = 'Cash from Operating Activity'
    assets_col = 'Total'
    
    if net_profit_col and cfo_col and assets_col:
        # Convert to numeric, errors='coerce' turns non-numeric into NaN
        np_val = pd.to_numeric(df[net_profit_col], errors='coerce')
        cfo_val = pd.to_numeric(df[cfo_col], errors='coerce')
        assets_val = pd.to_numeric(df[assets_col], errors='coerce')
        
        # Avoid division by zero
        assets_val = assets_val.replace(0, np.nan)
        features['Accruals_Ratio'] = (np_val - cfo_val) / assets_val
    else:
        print("Missing columns for Accruals Ratio.")
        features['Accruals_Ratio'] = np.nan
        
    # 2. Leverage Change (YoY change in Debt / Assets)
    debt_col = next((c for c in df.columns if 'Borrowings' in c or 'Debt' in c), None)
    if debt_col and assets_col:
        debt_val = pd.to_numeric(df[debt_col], errors='coerce')
        assets_val = pd.to_numeric(df[assets_col], errors='coerce')
        
        leverage = debt_val / assets_val.replace(0, np.nan)
        features['Leverage_Ratio'] = leverage
        # Group by company and calculate diff
        features['Leverage_Change_YoY'] = features.groupby('Company')['Leverage_Ratio'].diff()
    else:
        print("Missing columns for Leverage.")
        features['Leverage_Ratio'] = np.nan
        features['Leverage_Change_YoY'] = np.nan
        
    # 3. Margin Volatility (3-year rolling std of Operating Margin)
    sales_col = next((c for c in df.columns if 'Sales' in c or 'Revenue' in c), None)
    op_profit_col = next((c for c in df.columns if 'Operating Profit' in c or 'EBITDA' in c), None)
    
    if sales_col and op_profit_col:
        sales_val = pd.to_numeric(df[sales_col], errors='coerce')
        op_val = pd.to_numeric(df[op_profit_col], errors='coerce')
        
        margin = op_val / sales_val.replace(0, np.nan)
        features['Operating_Margin'] = margin
        
        # 3-year rolling std
        features['Margin_Volatility_3yr'] = features.groupby('Company')['Operating_Margin'].transform(lambda x: x.rolling(3, min_periods=2).std())
    else:
        print("Missing columns for Margin Volatility.")
        features['Operating_Margin'] = np.nan
        features['Margin_Volatility_3yr'] = np.nan
        
    # Drop rows with entirely NaN ML features (we need at least some data)
    ml_cols = ['Accruals_Ratio', 'Leverage_Change_YoY', 'Margin_Volatility_3yr']
    features.dropna(subset=ml_cols, how='all', inplace=True)
    
    features.to_csv(output_path, index=False)
    print(f"Feature engineering complete. Saved shape: {features.shape} to {output_path}")

if __name__ == "__main__":
    main()
