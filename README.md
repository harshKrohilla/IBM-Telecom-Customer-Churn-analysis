# Telecom Customer Churn — Data Analysis Project

## Project Description

This project performs a full end-to-end exploratory data analysis (EDA) on the **IBM Telco Customer Churn** dataset to identify the key drivers of customer churn in a telecommunications company and translate the findings into actionable business decisions.

The analysis follows five structured phases:
1. **Data Collection & Loading** — ingesting the CSV dataset into a pandas DataFrame.
2. **Data Quality Check** — detecting and correcting missing values, type mismatches, and duplicates.
3. **Grouping & Summarization** — segmenting customers by contract type, internet service, payment method, tenure, senior status, and add-on subscriptions.
4. **Data Visualization** — 8 comparative charts (pie, bar, KDE, heatmap) that make patterns immediately visible.
5. **Business Decisions** — seven concrete, data-backed recommendations to reduce churn.

---

## Dataset

| Field | Value |
|---|---|
| **File** | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| **Source** | [IBM Sample Data Sets — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Rows** | 7,043 customers |
| **Columns** | 21 (demographics, services, billing, churn label) |
| **Target variable** | `Churn` (Yes / No) |

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.9+ | Core programming language |
| pandas | Data loading, cleaning, grouping |
| numpy | Numerical computations |
| matplotlib | Base charting library |
| seaborn | Statistical visualization & heatmaps |
| Jupyter Notebook | Interactive development environment |

---

## Project Structure

```
telecomchurn/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
├── app.py                                  # ★ Streamlit interactive dashboard
├── Student_TelecomChurnAnalysis.py         # Main analysis script (static)
├── requirements.txt                        # Python dependencies
├── Student_TelecomChurnReport.docx         # Full project report (incl. Streamlit section)
├── README.md                               # This file
├── chart1_churn_distribution.png
├── chart2_churn_by_contract.png
├── chart3_churn_by_internet.png
├── chart4_monthly_charges_dist.png
├── chart5_churn_by_payment.png
├── chart6_churn_by_tenure.png
├── chart7_service_heatmap.png
└── chart8_senior_contract_churn.png
```

---

## Setup & Run Instructions

### 1. Prerequisites
- Python 3.9 or higher installed
- `pip` package manager

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. ★ Launch Streamlit Dashboard (recommended)
```bash
streamlit run app.py
```
Opens automatically at **http://localhost:8501** — interactive filters, live charts, KPI cards, and recommendations.

### 4. Run as a Python script (static analysis)
```bash
python Student_TelecomChurnAnalysis.py
```
Produces console output and saves 8 chart PNG files.

### 5. Run as a Jupyter Notebook
```bash
jupyter notebook
```
Then open `Student_TelecomChurnAnalysis.py` or paste the code into a new `.ipynb` notebook.

> **Note:** The CSV file must be in the same directory as the scripts.

---

## Key Findings

| Finding | Churn Rate |
|---|---|
| Month-to-month contract customers | **42.71%** |
| Fiber optic internet service customers | **41.89%** |
| Electronic check payment method | **45.29%** |
| Customers in first 12 months | **~47.7%** |
| Customers **without** OnlineSecurity | ~41% |
| Senior citizens | **41.68%** |

---

## Top Business Recommendations

1. **Incentivise contract upgrades** at months 3–5 for month-to-month subscribers.
2. **Audit fiber optic service quality** and introduce loyalty discounts.
3. **Promote auto-payment** (credit card / bank transfer) via bill credits.
4. **Launch a 90-day onboarding programme** for new customers.
5. **Bundle a free add-on service** (security/backup) for the first 3 months.
6. **Create simplified senior plans** with dedicated support.
7. **Introduce price-lock guarantees** for customers paying > $70/month.

---

## Author

**Student Name**  
Data Analysis Project — Telecom Customer Churn  
Dataset: IBM Telco Customer Churn (Kaggle)
