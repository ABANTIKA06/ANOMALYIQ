import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def main():
    input_path = 'data/ml_features.csv'
    output_path = 'data/anomaly_scores.csv'
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
        
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    
    ml_cols = ['Accruals_Ratio', 'Leverage_Change_YoY', 'Margin_Volatility_3yr']
    
    # Fill remaining NaNs with industry medians
    df[ml_cols] = df.groupby('Industry')[ml_cols].transform(lambda x: x.fillna(x.median()))
    
    # If still NaN (e.g. whole industry is NaN), fill with global median
    df[ml_cols] = df[ml_cols].fillna(df[ml_cols].median())
    
    df['Anomaly_Score'] = np.nan
    df['Is_Anomaly'] = -1
    
    # We train isolation forest per industry to normalize for industry norms
    for industry, group in df.groupby('Industry'):
        if len(group) > 10:
            X = group[ml_cols]
            
            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            clf = IsolationForest(contamination=0.05, random_state=42)
            
            # Fit and predict
            clf.fit(X_scaled)
            
            # Scores (lower is more anomalous, we multiply by -1 so higher = more anomalous)
            scores = clf.decision_function(X_scaled) * -1
            preds = clf.predict(X_scaled) # -1 is anomaly, 1 is normal
            
            df.loc[group.index, 'Anomaly_Score'] = scores
            df.loc[group.index, 'Is_Anomaly'] = (preds == -1).astype(int)
        else:
            # Not enough data for this industry, fallback to global
            pass

    # For any rows not scored (small industries), score globally
    unscored = df['Anomaly_Score'].isna()
    if unscored.any():
        X = df.loc[unscored, ml_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = IsolationForest(contamination=0.05, random_state=42)
        clf.fit(X_scaled)
        df.loc[unscored, 'Anomaly_Score'] = clf.decision_function(X_scaled) * -1
        df.loc[unscored, 'Is_Anomaly'] = (clf.predict(X_scaled) == -1).astype(int)

    # Save results
    df.sort_values(by='Anomaly_Score', ascending=False, inplace=True)
    df.to_csv(output_path, index=False)
    
    top_anomalies = df[df['Is_Anomaly'] == 1].groupby('Company').size().sort_values(ascending=False).head(10)
    print("\nTop 10 most frequently flagged companies:")
    print(top_anomalies)
    
    print(f"\nModel training complete. Saved to {output_path}")

if __name__ == "__main__":
    main()
