import pandas as pd
import os

def main():
    input_path = 'data/anomaly_scores.csv'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
        
    df = pd.read_csv(input_path)
    
    anomalies = df[df['Is_Anomaly'] == 1].copy()
    
    # We want to see which companies show up multiple times or have the highest anomaly scores
    top_scores = anomalies.sort_values(by='Anomaly_Score', ascending=False).head(20)
    
    report = "# Top Financial Anomalies Report\n\n"
    report += "This report lists the most severe financial anomalies detected by the Isolation Forest model.\n\n"
    
    report += "## Top 20 Most Anomalous Company-Years\n\n"
    report += "| Company | Year | Industry | Anomaly Score | Accruals Ratio | Leverage YoY | Margin Volatility |\n"
    report += "|---------|------|----------|---------------|----------------|--------------|-------------------|\n"
    
    for idx, row in top_scores.iterrows():
        comp = row['Company']
        yr = row['Year']
        ind = row['Industry']
        score = f"{row['Anomaly_Score']:.3f}"
        accr = f"{row['Accruals_Ratio']:.3f}"
        lev = f"{row['Leverage_Change_YoY']:.3f}"
        marg = f"{row['Margin_Volatility_3yr']:.3f}"
        
        report += f"| {comp} | {yr} | {ind} | {score} | {accr} | {lev} | {marg} |\n"
        
    report += "\n\n## Analysis\n"
    report += "- **Accruals Ratio**: High positive values mean reported profit is much higher than actual cash generated. (Classic red flag)\n"
    report += "- **Leverage YoY**: Massive spikes in debt relative to assets.\n"
    report += "- **Margin Volatility**: Extreme swings in operating margins compared to industry peers.\n"
    
    with open('notebooks/results_analysis.md', 'w') as f:
        f.write(report)
        
    print("Report generated at notebooks/results_analysis.md")

if __name__ == "__main__":
    main()
