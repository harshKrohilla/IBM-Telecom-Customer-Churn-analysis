# ============================================================
# Student_TelecomChurnAnalysis.py
# Telecom Customer Churn — Data Analysis Project
# ============================================================

# ─── 0. IMPORTS ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120

# ─── 1. COLLECT AND LOAD DATASET ────────────────────────────
print("=" * 60)
print("STEP 1 — COLLECT AND LOAD DATASET")
print("=" * 60)

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(f"\nDataset shape : {df.shape[0]} rows × {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)

# ─── 2. CHECK FOR MISSING OR INCORRECT VALUES ───────────────
print("\n" + "=" * 60)
print("STEP 2 — DATA QUALITY CHECK")
print("=" * 60)

# 2a. Missing values
print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "No null values found.")

# 2b. TotalCharges is stored as object — coerce to numeric
print("\n--- Converting TotalCharges to numeric ---")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
newly_null = df["TotalCharges"].isnull().sum()
print(f"Rows with blank TotalCharges (new NaN): {newly_null}")

# Fill blanks with 0 (new customers with 0 tenure)
df["TotalCharges"].fillna(0, inplace=True)
print("Blank TotalCharges filled with 0.")

# 2c. SeniorCitizen: 0/1 → human-readable label
df["SeniorCitizenLabel"] = df["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})

# 2d. Churn: Yes/No → binary flag
df["ChurnFlag"] = df["Churn"].map({"Yes": 1, "No": 0})

# 2e. Duplicate check
dupes = df.duplicated().sum()
print(f"\nDuplicate rows: {dupes}")

# 2f. Summary statistics after cleaning
print("\n--- Numeric Summary After Cleaning ---")
print(df[["tenure", "MonthlyCharges", "TotalCharges"]].describe().round(2))

# 2g. Unique value counts for key categoricals
print("\n--- Key Categorical Distributions ---")
for col in ["gender", "Contract", "InternetService", "PaymentMethod", "Churn"]:
    print(f"\n{col}:\n{df[col].value_counts()}")

# ─── 3. GROUP AND SUMMARIZE ──────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — GROUP AND SUMMARIZE")
print("=" * 60)

# 3a. Overall churn rate
overall_churn = df["ChurnFlag"].mean() * 100
print(f"\nOverall churn rate : {overall_churn:.2f}%")
print(f"Churned customers  : {df['ChurnFlag'].sum()} / {len(df)}")

# 3b. Churn by Contract type
churn_contract = (
    df.groupby("Contract")["ChurnFlag"]
    .agg(["count", "sum", "mean"])
    .rename(columns={"count": "Total", "sum": "Churned", "mean": "ChurnRate"})
)
churn_contract["ChurnRate%"] = (churn_contract["ChurnRate"] * 100).round(2)
print("\n--- Churn by Contract Type ---")
print(churn_contract.drop(columns="ChurnRate"))

# 3c. Churn by Internet Service
churn_internet = (
    df.groupby("InternetService")["ChurnFlag"]
    .agg(["count", "sum", "mean"])
    .rename(columns={"count": "Total", "sum": "Churned", "mean": "ChurnRate"})
)
churn_internet["ChurnRate%"] = (churn_internet["ChurnRate"] * 100).round(2)
print("\n--- Churn by Internet Service ---")
print(churn_internet.drop(columns="ChurnRate"))

# 3d. Churn by Payment Method
churn_payment = (
    df.groupby("PaymentMethod")["ChurnFlag"]
    .agg(["count", "sum", "mean"])
    .rename(columns={"count": "Total", "sum": "Churned", "mean": "ChurnRate"})
)
churn_payment["ChurnRate%"] = (churn_payment["ChurnRate"] * 100).round(2)
print("\n--- Churn by Payment Method ---")
print(churn_payment.drop(columns="ChurnRate"))

# 3e. Churn by Gender
churn_gender = df.groupby("gender")["ChurnFlag"].mean().mul(100).round(2)
print("\n--- Churn Rate by Gender ---")
print(churn_gender)

# 3f. Churn by Senior Citizen
churn_senior = df.groupby("SeniorCitizenLabel")["ChurnFlag"].mean().mul(100).round(2)
print("\n--- Churn Rate by Senior Citizen ---")
print(churn_senior)

# 3g. Average Monthly & Total Charges — Churn vs Retained
charges_summary = df.groupby("Churn")[["MonthlyCharges", "TotalCharges", "tenure"]].mean().round(2)
print("\n--- Average Charges & Tenure — Churned vs Retained ---")
print(charges_summary)

# 3h. Tenure segmentation
bins = [0, 12, 24, 48, 72]
labels = ["0–12 mo", "13–24 mo", "25–48 mo", "49–72 mo"]
df["TenureGroup"] = pd.cut(df["tenure"], bins=bins, labels=labels)
churn_tenure = df.groupby("TenureGroup")["ChurnFlag"].mean().mul(100).round(2)
print("\n--- Churn Rate by Tenure Group ---")
print(churn_tenure)

# ─── 4. CREATE CHARTS ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — VISUALIZATIONS")
print("=" * 60)

# ── Chart 1: Overall Churn Distribution (Pie) ──────────────
fig1, ax1 = plt.subplots(figsize=(6, 5))
churn_counts = df["Churn"].value_counts()
colors = ["#3b82d4", "#ef4444"]
wedge_props = {"linewidth": 2, "edgecolor": "white"}
ax1.pie(
    churn_counts,
    labels=churn_counts.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=colors,
    wedgeprops=wedge_props,
    textprops={"fontsize": 13},
)
ax1.set_title("Overall Customer Churn Distribution", fontweight="bold", fontsize=14)
plt.tight_layout()
plt.savefig("chart1_churn_distribution.png")
plt.show()
print("Chart 1 saved → chart1_churn_distribution.png")

# ── Chart 2: Churn Rate by Contract Type (Bar) ─────────────
fig2, ax2 = plt.subplots(figsize=(7, 5))
contract_order = ["Month-to-month", "One year", "Two year"]
rates = churn_contract.loc[contract_order, "ChurnRate%"]
bars = ax2.bar(contract_order, rates, color=["#ef4444", "#f59e0b", "#22c55e"], width=0.5, edgecolor="white")
for bar, val in zip(bars, rates):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{val}%", ha="center", va="bottom", fontweight="bold")
ax2.set_title("Churn Rate by Contract Type", fontweight="bold", fontsize=14)
ax2.set_ylabel("Churn Rate (%)")
ax2.set_ylim(0, rates.max() + 8)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.savefig("chart2_churn_by_contract.png")
plt.show()
print("Chart 2 saved → chart2_churn_by_contract.png")

# ── Chart 3: Churn Rate by Internet Service ────────────────
fig3, ax3 = plt.subplots(figsize=(7, 5))
inet_rates = churn_internet["ChurnRate%"].sort_values(ascending=False)
bars3 = ax3.barh(inet_rates.index, inet_rates.values,
                 color=["#ef4444", "#3b82d4", "#a3e635"], edgecolor="white")
for bar, val in zip(bars3, inet_rates.values):
    ax3.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
             f"{val}%", va="center", fontweight="bold")
ax3.set_title("Churn Rate by Internet Service Type", fontweight="bold", fontsize=14)
ax3.set_xlabel("Churn Rate (%)")
ax3.xaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.savefig("chart3_churn_by_internet.png")
plt.show()
print("Chart 3 saved → chart3_churn_by_internet.png")

# ── Chart 4: Monthly Charges Distribution — Churn vs No ───
fig4, ax4 = plt.subplots(figsize=(8, 5))
df[df["Churn"] == "Yes"]["MonthlyCharges"].plot.kde(ax=ax4, label="Churned", color="#ef4444", linewidth=2.5)
df[df["Churn"] == "No"]["MonthlyCharges"].plot.kde(ax=ax4, label="Retained", color="#3b82d4", linewidth=2.5)
ax4.set_title("Monthly Charges Distribution: Churned vs Retained", fontweight="bold", fontsize=14)
ax4.set_xlabel("Monthly Charges ($)")
ax4.set_ylabel("Density")
ax4.legend(fontsize=12)
plt.tight_layout()
plt.savefig("chart4_monthly_charges_dist.png")
plt.show()
print("Chart 4 saved → chart4_monthly_charges_dist.png")

# ── Chart 5: Churn Rate by Payment Method ─────────────────
fig5, ax5 = plt.subplots(figsize=(9, 5))
pm_rates = churn_payment["ChurnRate%"].sort_values(ascending=True)
colors5 = ["#22c55e", "#a3e635", "#f59e0b", "#ef4444"]
bars5 = ax5.barh(pm_rates.index, pm_rates.values, color=colors5, edgecolor="white")
for bar, val in zip(bars5, pm_rates.values):
    ax5.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
             f"{val}%", va="center", fontweight="bold")
ax5.set_title("Churn Rate by Payment Method", fontweight="bold", fontsize=14)
ax5.set_xlabel("Churn Rate (%)")
ax5.xaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.savefig("chart5_churn_by_payment.png")
plt.show()
print("Chart 5 saved → chart5_churn_by_payment.png")

# ── Chart 6: Churn Rate by Tenure Group ───────────────────
fig6, ax6 = plt.subplots(figsize=(8, 5))
bars6 = ax6.bar(churn_tenure.index, churn_tenure.values,
                color=["#ef4444", "#f59e0b", "#3b82d4", "#22c55e"],
                width=0.5, edgecolor="white")
for bar, val in zip(bars6, churn_tenure.values):
    ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
             f"{val}%", ha="center", fontweight="bold")
ax6.set_title("Churn Rate by Tenure Group", fontweight="bold", fontsize=14)
ax6.set_ylabel("Churn Rate (%)")
ax6.set_ylim(0, churn_tenure.max() + 8)
ax6.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.savefig("chart6_churn_by_tenure.png")
plt.show()
print("Chart 6 saved → chart6_churn_by_tenure.png")

# ── Chart 7: Heatmap — Churn vs Add-on Services ───────────
services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"]

# Keep only Yes/No (drop "No internet service")
service_churn = {}
for s in services:
    temp = df[df[s].isin(["Yes", "No"])]
    service_churn[s] = temp.groupby(s)["ChurnFlag"].mean().mul(100).round(2)

heatmap_df = pd.DataFrame(service_churn).T  # rows=service, cols=Yes/No
fig7, ax7 = plt.subplots(figsize=(7, 5))
sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="RdYlGn_r",
            linewidths=0.5, linecolor="white", ax=ax7,
            cbar_kws={"label": "Churn Rate (%)"})
ax7.set_title("Churn Rate (%) by Add-on Service Subscription", fontweight="bold", fontsize=13)
ax7.set_xlabel("Subscribed to Service")
plt.tight_layout()
plt.savefig("chart7_service_heatmap.png")
plt.show()
print("Chart 7 saved → chart7_service_heatmap.png")

# ── Chart 8: Senior vs Non-Senior Churn (Grouped Bar) ─────
fig8, ax8 = plt.subplots(figsize=(7, 5))
senior_contract = (
    df.groupby(["SeniorCitizenLabel", "Contract"])["ChurnFlag"]
    .mean()
    .mul(100)
    .unstack()
)
senior_contract.plot(kind="bar", ax=ax8, edgecolor="white",
                     color=["#ef4444", "#f59e0b", "#22c55e"])
ax8.set_title("Churn Rate by Senior Status & Contract Type", fontweight="bold", fontsize=14)
ax8.set_ylabel("Churn Rate (%)")
ax8.set_xlabel("")
ax8.yaxis.set_major_formatter(mtick.PercentFormatter())
ax8.legend(title="Contract", fontsize=10)
ax8.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig("chart8_senior_contract_churn.png")
plt.show()
print("Chart 8 saved → chart8_senior_contract_churn.png")

# ─── 5. BUSINESS DECISIONS ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — BUSINESS INSIGHTS & RECOMMENDATIONS")
print("=" * 60)

insights = [
    ("1. High-Risk Segment: Month-to-Month Customers",
     f"  Churn rate = {churn_contract.loc['Month-to-month', 'ChurnRate%']}% vs "
     f"{churn_contract.loc['Two year', 'ChurnRate%']}% for two-year contracts.\n"
     "  → RECOMMENDATION: Offer incentivised annual upgrade bundles at month 3–5 of tenure."),

    ("2. Fiber Optic Subscribers Churn Most",
     f"  Fiber optic churn rate = {churn_internet.loc['Fiber optic', 'ChurnRate%']}%.\n"
     "  → RECOMMENDATION: Audit fiber service quality, address latency/price concerns, "
     "and provide loyalty discounts at 6-month mark."),

    ("3. Electronic Check Payers Are Highest-Risk",
     f"  Electronic check churn = {churn_payment['ChurnRate%'].max()}%.\n"
     "  → RECOMMENDATION: Promote auto-payment (credit card / bank transfer) via bill-credit "
     "incentives — auto-pay customers churn far less."),

    ("4. Early-Tenure Customers (0–12 mo) Need Retention Focus",
     f"  Churn rate in first year = {churn_tenure['0–12 mo']}%.\n"
     "  → RECOMMENDATION: Implement a 90-day onboarding programme with proactive "
     "support calls, service health checks, and welcome discounts."),

    ("5. Add-On Services Act as Churn Shields",
     "  Customers without OnlineSecurity, TechSupport, or OnlineBackup churn at ~2× the rate.\n"
     "  → RECOMMENDATION: Bundle 1 free security/backup add-on for the first 3 months "
     "to increase stickiness."),

    ("6. Senior Citizens Are Under-Served",
     f"  Senior citizen churn = {churn_senior['Senior']}% vs {churn_senior['Non-Senior']}%.\n"
     "  → RECOMMENDATION: Create simplified plans + dedicated support tier for senior "
     "customers; partner with community programmes for visibility."),

    ("7. High Monthly Charges Correlate with Churn",
     f"  Churned avg monthly charge = ${charges_summary.loc['Yes', 'MonthlyCharges']:.2f} vs "
     f"${charges_summary.loc['No', 'MonthlyCharges']:.2f} for retained customers.\n"
     "  → RECOMMENDATION: Introduce mid-tier plans and price-lock guarantees for customers "
     "spending >$70/month to reduce price-driven exits."),
]

for title, body in insights:
    print(f"\n{'─'*55}")
    print(title)
    print(body)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
