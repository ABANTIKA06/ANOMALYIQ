# AnomalyIQ · Nifty 500 Corporate Financial Anomaly Intelligence System

An end-to-end Machine Learning anomaly detection system and interactive Plotly Dash visualization dashboard for **492 Nifty 500 companies** spanning 10 fiscal years (FY2015–FY2024).

---

## 📁 Standalone Project Folder Structure (`StockAnalysis/ANOMALYIQ/`)

```
StockAnalysis/
└── ANOMALYIQ/                  # Dedicated Subfolder for Anomaly Detection Project
    ├── app.py                  # Main Plotly Dash Dashboard Entrypoint
    ├── requirements.txt        # Python dependencies
    ├── README.md               # System Architecture & Documentation
    ├── .env / .env.example     # Environment configuration & credentials
    │
    ├── src/                    # Core Python Pipeline & Scraper Package
    │   ├── data_ingestion.py   # Screener.in Excel parser & panel builder
    │   ├── feature_engineering.py # Financial ratio feature calculation engine
    │   ├── model_training.py   # Isolation Forest machine learning model
    │   ├── generate_report.py  # Automated report generator
    │   ├── screener_downloader.py # Automated Screener.in Playwright scraper
    │   └── screener_retry_failed.py # Failure recovery scraper script
    │
    ├── scripts/                # Pipeline Helper Scripts
    │   ├── add_industry.py     # Sector enrichment script
    │   └── generate_notebook.py# Automated notebook generator
    │
    ├── data/                   # Financial Datasets & ML Model Outputs
    │   ├── anomaly_scores.csv  # Isolation Forest anomaly scores per company-year
    │   ├── raw_panel_data.csv  # Merged 10-year P&L, Balance Sheet, Cash Flow panel
    │   ├── ml_features.csv     # Computed financial ratio features
    │   ├── ind_nifty500list.csv# Official Nifty 500 company master list
    │   └── raw_exports/        # Company Excel export files
    │
    ├── screener_exports/       # Bulk Screener.in Excel files (492 companies)
    │
    ├── notebooks/              # Google Colab & Jupyter Notebooks
    │   ├── Colab_ML_Pipeline.ipynb # Standalone Google Colab ML notebook
    │   └── results_analysis.md # Statistical summary of model results
    │
    ├── assets/                 # Design references & screenshots
    │   └── website_style.png   # Portfolio design reference
    │
    └── logs/                   # Execution & Downloader Log Files
        ├── screener_downloader.log
        ├── screener_retry.log
        └── download_log.csv
```

---

## 🎨 Visualization Features

1. **Light & Dark Theme Variations**:
   - Integrated **Theme Toggle** in the top navigation header (`☀️ LIGHT MODE` / `🌙 DARK MODE`).
   - High-contrast dark theme with styled dropdown controls (`#14171F` dark surface, `#F8F9FA` bright white text).

2. **24-Color High-Contrast Scatter Canvas**:
   - 24-color vibrant palette for the *Multidimensional Feature Canvas* graph.
   - Distinct contrast outlines (`#FFFFFF` in light mode, `#14171F` in dark mode) so overlapping data points remain visible.

3. **4 Interactive Visualization Views**:
   - `01 INDUSTRY OVERVIEW`: Aggregate sector anomaly rate & score distributions.
   - `02 ANOMALY EXPLORER`: 2D multi-dimensional feature scatter with hover tooltips & top 20 outlier table.
   - `03 COMPANY DEEP DIVE`: 10-year longitudinal trajectory of Revenue, Profit, Cash Flow, Debt, and Anomaly timeline.
   - `04 METHODOLOGY`: Financial theory breakdown (Sloan's Accruals, Debt YoY, Isolation Forest math).

---

## 🚀 Quick Start

### 1. Navigate to Project Subfolder
```powershell
cd ANOMALYIQ
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Launch Dashboard
```powershell
python app.py
```
Open your browser at `http://127.0.0.1:8050` to interact with the dashboard.
