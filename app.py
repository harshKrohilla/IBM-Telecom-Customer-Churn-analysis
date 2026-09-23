# ============================================================
# app.py  —  Telecom Customer Churn  |  Streamlit Dashboard
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Analysis",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS  — clean card / metric styling
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── global ─────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: "Segoe UI", system-ui, sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; }

/* ── page title ──────────────────────────────────────────── */
.page-title {
    font-size: 2rem; font-weight: 700; color: #1f2328;
    border-left: 5px solid #3b82f6; padding-left: 14px;
    margin-bottom: 0.2rem;
}
.page-sub { font-size: 0.95rem; color: #57606a; margin-bottom: 1.5rem; }

/* ── section header ──────────────────────────────────────── */
.section-header {
    font-size: 1.15rem; font-weight: 600; color: #1f2328;
    border-bottom: 2px solid #e5e7eb; padding-bottom: 6px;
    margin-top: 1.6rem; margin-bottom: 0.8rem;
}

/* ── KPI card ────────────────────────────────────────────── */
.kpi-card {
    background: #f7f8fa; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 18px 20px;
    text-align: center;
}
.kpi-label { font-size: 0.80rem; color: #57606a; text-transform: uppercase;
             letter-spacing: .05em; margin-bottom: 6px; }
.kpi-value { font-size: 2rem; font-weight: 700; color: #1f2328; line-height: 1; }
.kpi-delta { font-size: 0.78rem; margin-top: 4px; }
.kpi-red   { color: #ef4444; }
.kpi-green { color: #22c55e; }
.kpi-blue  { color: #3b82f6; }

/* ── insight box ─────────────────────────────────────────── */
.insight-box {
    background: #f0f7ff; border-left: 4px solid #3b82f6;
    border-radius: 6px; padding: 14px 18px; margin-bottom: 10px;
}
.insight-title { font-weight: 600; color: #1f2328; margin-bottom: 4px; }
.insight-body  { font-size: 0.92rem; color: #374151; }

/* ── warning box ─────────────────────────────────────────── */
.warn-box {
    background: #fff7ed; border-left: 4px solid #f59e0b;
    border-radius: 6px; padding: 14px 18px; margin-bottom: 10px;
}

/* ── recommendation card ─────────────────────────────────── */
.rec-card {
    background: #ffffff; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 16px 20px; margin-bottom: 12px;
}
.rec-num  { font-size: 1.4rem; font-weight: 700; color: #3b82f6; }
.rec-head { font-size: 1rem; font-weight: 600; color: #1f2328; }
.rec-body { font-size: 0.90rem; color: #374151; margin-top: 6px; line-height: 1.55; }
.rec-action { font-size: 0.88rem; color: #16a34a; font-weight: 600;
              margin-top: 6px; }

/* ── tab strip ───────────────────────────────────────────── */
div[data-baseweb="tab-list"] { gap: 4px; }
div[data-baseweb="tab"] { border-radius: 6px 6px 0 0 !important; }

/* ── dataframe ───────────────────────────────────────────── */
.dataframe { font-size: 0.88rem !important; }

/* ── footer ──────────────────────────────────────────────── */
.footer { text-align: center; font-size: 0.78rem; color: #adb5bd;
          border-top: 1px solid #e5e7eb; margin-top: 3rem; padding-top: 12px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING  (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["ChurnFlag"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["SeniorCitizenLabel"] = df["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})
    bins   = [0, 12, 24, 48, 72]
    labels = ["0–12 mo", "13–24 mo", "25–48 mo", "49–72 mo"]
    df["TenureGroup"] = pd.cut(df["tenure"], bins=bins, labels=labels)
    return df


# ─────────────────────────────────────────────────────────────
# SIDEBAR — file upload + filters
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/antenna.png", width=56)
    st.markdown("## 📡 Telecom Churn")
    st.markdown("---")

    uploaded = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded:
        df_raw = load_data(uploaded)
        data_source = f"Uploaded: {uploaded.name}"
    else:
        df_raw = load_data("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        data_source = "Default: IBM Telco Churn Dataset"

    st.markdown(f"<small style='color:#57606a'>📂 {data_source}</small>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🔧 Filters")
    sel_gender   = st.multiselect("Gender",           df_raw["gender"].unique(),
                                  default=list(df_raw["gender"].unique()))
    sel_senior   = st.multiselect("Senior Citizen",   df_raw["SeniorCitizenLabel"].unique(),
                                  default=list(df_raw["SeniorCitizenLabel"].unique()))
    sel_contract = st.multiselect("Contract Type",    df_raw["Contract"].unique(),
                                  default=list(df_raw["Contract"].unique()))
    sel_internet = st.multiselect("Internet Service", df_raw["InternetService"].unique(),
                                  default=list(df_raw["InternetService"].unique()))

    st.markdown("---")
    st.markdown("<small style='color:#adb5bd'>Telecom Churn Analysis · IBM Dataset</small>",
                unsafe_allow_html=True)

# apply filters
df = df_raw[
    df_raw["gender"].isin(sel_gender) &
    df_raw["SeniorCitizenLabel"].isin(sel_senior) &
    df_raw["Contract"].isin(sel_contract) &
    df_raw["InternetService"].isin(sel_internet)
].copy()

if df.empty:
    st.error("⚠️ No data matches the selected filters. Please adjust the sidebar filters.")
    st.stop()


# ─────────────────────────────────────────────────────────────
# PRE-COMPUTE AGGREGATES  (used across tabs)
# ─────────────────────────────────────────────────────────────
total_customers  = len(df)
total_churned    = int(df["ChurnFlag"].sum())
overall_churn    = df["ChurnFlag"].mean() * 100
avg_monthly      = df["MonthlyCharges"].mean()
avg_tenure       = df["tenure"].mean()
revenue_at_risk  = df[df["Churn"] == "Yes"]["MonthlyCharges"].sum()

churn_contract = (df.groupby("Contract")["ChurnFlag"]
                  .agg(Total="count", Churned="sum", ChurnRate="mean")
                  .assign(**{"Churn Rate (%)": lambda x: (x["ChurnRate"]*100).round(2)})
                  .drop(columns="ChurnRate"))

churn_internet = (df.groupby("InternetService")["ChurnFlag"]
                  .agg(Total="count", Churned="sum", ChurnRate="mean")
                  .assign(**{"Churn Rate (%)": lambda x: (x["ChurnRate"]*100).round(2)})
                  .drop(columns="ChurnRate"))

churn_payment = (df.groupby("PaymentMethod")["ChurnFlag"]
                 .agg(Total="count", Churned="sum", ChurnRate="mean")
                 .assign(**{"Churn Rate (%)": lambda x: (x["ChurnRate"]*100).round(2)})
                 .drop(columns="ChurnRate"))

churn_gender = df.groupby("gender")["ChurnFlag"].mean().mul(100).round(2)
churn_senior = df.groupby("SeniorCitizenLabel")["ChurnFlag"].mean().mul(100).round(2)
charges_summary = df.groupby("Churn")[["MonthlyCharges", "TotalCharges", "tenure"]].mean().round(2)
churn_tenure = df.groupby("TenureGroup", observed=True)["ChurnFlag"].mean().mul(100).round(2)

services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"]
service_churn = {}
for s in services:
    tmp = df[df[s].isin(["Yes", "No"])]
    service_churn[s] = tmp.groupby(s)["ChurnFlag"].mean().mul(100).round(2)
heatmap_df = pd.DataFrame(service_churn).T


# ─────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────
C_RED    = "#ef4444"
C_GREEN  = "#22c55e"
C_BLUE   = "#3b82f6"
C_AMBER  = "#f59e0b"
C_PURPLE = "#7c5cd8"
C_LIME   = "#84cc16"
PALETTE  = [C_BLUE, C_RED, C_GREEN, C_AMBER, C_PURPLE, C_LIME]


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
TAB_ICONS = ["🏠 Overview", "🔍 Data Quality", "📊 Group Analysis",
             "📈 Charts", "💡 Business Insights"]
t1, t2, t3, t4, t5 = st.tabs(TAB_ICONS)


# ══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW  (Step 1: Collect & Load)
# ══════════════════════════════════════════════════════════════
with t1:
    st.markdown('<div class="page-title">📡 Telecom Customer Churn Dashboard</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-sub">IBM Telco Customer Churn Dataset — Exploratory Data Analysis</div>',
                unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (k1, "Total Customers",   f"{total_customers:,}",  "", "kpi-blue"),
        (k2, "Churned Customers", f"{total_churned:,}",    "⚠ High-risk", "kpi-red"),
        (k3, "Churn Rate",        f"{overall_churn:.1f}%", "Industry avg ~20%", "kpi-red"),
        (k4, "Avg Monthly Bill",  f"${avg_monthly:.2f}",   "", "kpi-blue"),
        (k5, "Avg Tenure",        f"{avg_tenure:.1f} mo",  "", "kpi-blue"),
    ]
    for col, label, value, delta, cls in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value {cls}">{value}</div>
              <div class="kpi-delta kpi-red">{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Revenue at risk callout ───────────────────────────────
    st.markdown(f"""
    <div class="warn-box">
      <strong>💰 Monthly Revenue at Risk:</strong> &nbsp;
      <span style="font-size:1.1rem;font-weight:700;color:#d97706">${revenue_at_risk:,.0f}</span>
      &nbsp; — total monthly charges from churned customers.
    </div>""", unsafe_allow_html=True)

    # ── Dataset preview ───────────────────────────────────────
    st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.dataframe(df.head(10), use_container_width=True, height=320)
    with col_r:
        st.markdown("**Dataset Info**")
        info = {
            "Rows":    total_customers,
            "Columns": len(df.columns),
            "Churned": total_churned,
            "Retained": total_customers - total_churned,
            "Churn Rate": f"{overall_churn:.2f}%",
        }
        for k, v in info.items():
            st.metric(k, v)

    # ── Column overview ───────────────────────────────────────
    st.markdown('<div class="section-header">🗂 Column Descriptions</div>', unsafe_allow_html=True)
    col_meta = pd.DataFrame({
        "Column": ["customerID","gender","SeniorCitizen","Partner","Dependents",
                   "tenure","PhoneService","MultipleLines","InternetService",
                   "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
                   "StreamingTV","StreamingMovies","Contract","PaperlessBilling",
                   "PaymentMethod","MonthlyCharges","TotalCharges","Churn"],
        "Type":   ["ID","Categorical","Binary","Categorical","Categorical",
                   "Numeric","Categorical","Categorical","Categorical",
                   "Categorical","Categorical","Categorical","Categorical",
                   "Categorical","Categorical","Categorical","Categorical",
                   "Categorical","Numeric","Numeric","Target"],
        "Description": [
            "Unique customer identifier",
            "Male / Female",
            "1 = Senior citizen, 0 = Not",
            "Has a partner (Yes/No)",
            "Has dependents (Yes/No)",
            "Months the customer has been with the company",
            "Phone service subscription",
            "Multiple phone lines",
            "DSL / Fiber optic / No",
            "Online security add-on",
            "Online backup add-on",
            "Device protection add-on",
            "Tech support add-on",
            "Streaming TV add-on",
            "Streaming movies add-on",
            "Month-to-month / One year / Two year",
            "Uses paperless billing",
            "Electronic check / Mailed check / Bank transfer / Credit card",
            "Monthly billing amount ($)",
            "Total billing amount ($)",
            "⭐ TARGET — Yes = churned, No = retained",
        ]
    })
    st.dataframe(col_meta, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — DATA QUALITY  (Step 2: Check for issues)
# ══════════════════════════════════════════════════════════════
with t2:
    st.markdown('<div class="page-title">🔍 Data Quality Check</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Missing values, data types, duplicates, and statistical summary.</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        missing_count = df_raw.isnull().sum().sum()
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Missing Values</div>
          <div class="kpi-value {'kpi-red' if missing_count else 'kpi-green'}">{missing_count}</div>
          <div class="kpi-delta">{'⚠ Requires handling' if missing_count else '✓ None found'}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        dupe_count = df_raw.duplicated().sum()
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Duplicate Rows</div>
          <div class="kpi-value {'kpi-red' if dupe_count else 'kpi-green'}">{dupe_count}</div>
          <div class="kpi-delta">{'⚠ Found duplicates' if dupe_count else '✓ No duplicates'}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        tc_blanks = (pd.to_numeric(df_raw["TotalCharges"], errors="coerce").isna()).sum()
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">TotalCharges Blanks</div>
          <div class="kpi-value kpi-amber" style="color:#f59e0b">{tc_blanks}</div>
          <div class="kpi-delta">Coerced → filled with 0</div>
        </div>""", unsafe_allow_html=True)

    # ── Fixes applied ─────────────────────────────────────────
    st.markdown('<div class="section-header">🛠 Data Fixes Applied</div>', unsafe_allow_html=True)
    fixes = [
        ("TotalCharges — Type Correction",
         f"Column was stored as `object` (string). {tc_blanks} blank entries coerced to NaN, then filled with 0 (new customers with tenure = 0)."),
        ("SeniorCitizen — Label Mapping",
         "Binary 0/1 integer mapped to human-readable 'Non-Senior' / 'Senior' labels for readability in charts."),
        ("Churn — Binary Flag",
         "Yes/No text mapped to ChurnFlag (1/0) to enable numeric aggregation (mean = churn rate)."),
        ("TenureGroup — Segmentation",
         "Continuous tenure bucketed into 4 groups: 0–12 mo, 13–24 mo, 25–48 mo, 49–72 mo."),
    ]
    for title, body in fixes:
        st.markdown(f"""
        <div class="insight-box">
          <div class="insight-title">✅ {title}</div>
          <div class="insight-body">{body}</div>
        </div>""", unsafe_allow_html=True)

    # ── Data types ────────────────────────────────────────────
    st.markdown('<div class="section-header">🗃 Data Types</div>', unsafe_allow_html=True)
    dtype_df = df_raw.dtypes.reset_index()
    dtype_df.columns = ["Column", "Original dtype"]
    dtype_df["After Cleaning"] = dtype_df["Column"].map(lambda c: str(df[c].dtype) if c in df.columns else "—")
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    # ── Numeric summary ───────────────────────────────────────
    st.markdown('<div class="section-header">📐 Numeric Summary Statistics</div>', unsafe_allow_html=True)
    num_summary = df[["tenure", "MonthlyCharges", "TotalCharges"]].describe().T.round(2)
    num_summary.index.name = "Column"
    st.dataframe(num_summary, use_container_width=True)

    # ── Distributions (after cleaning) ───────────────────────
    st.markdown('<div class="section-header">📊 Value Distributions (Categorical Columns)</div>',
                unsafe_allow_html=True)
    cats = ["gender", "Contract", "InternetService", "PaymentMethod", "Churn"]
    cols_dist = st.columns(len(cats))
    for i, col_name in enumerate(cats):
        with cols_dist[i]:
            vc = df[col_name].value_counts()
            fig = px.bar(x=vc.index, y=vc.values,
                         labels={"x": col_name, "y": "Count"},
                         color=vc.index,
                         color_discrete_sequence=PALETTE,
                         height=260)
            fig.update_layout(showlegend=False, margin=dict(t=30, b=30, l=20, r=20),
                               title=dict(text=col_name, font_size=13),
                               xaxis_tickangle=-25)
            fig.update_traces(texttemplate="%{y:,}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — GROUP ANALYSIS  (Step 3: Group & Summarize)
# ══════════════════════════════════════════════════════════════
with t3:
    st.markdown('<div class="page-title">📊 Group & Summarize</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Churn rates segmented by key customer dimensions.</div>',
                unsafe_allow_html=True)

    # ── Overall churn rate gauge ──────────────────────────────
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=overall_churn,
        delta={"reference": 20, "suffix": "%", "valueformat": ".1f"},
        number={"suffix": "%", "font": {"size": 46, "color": "#1f2328"}},
        gauge={
            "axis": {"range": [0, 60], "tickwidth": 1, "tickcolor": "#e5e7eb"},
            "bar":  {"color": C_RED, "thickness": 0.25},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 15],  "color": "#dcfce7"},
                {"range": [15, 30], "color": "#fef9c3"},
                {"range": [30, 60], "color": "#fee2e2"},
            ],
            "threshold": {"line": {"color": "#f59e0b", "width": 4},
                          "thickness": 0.8, "value": 20},
        },
        title={"text": "Overall Churn Rate<br><span style='font-size:0.85em;color:#57606a'>Industry target &lt;20%</span>",
               "font": {"size": 18}},
    ))
    gauge.update_layout(height=260, margin=dict(t=20, b=10, l=30, r=30), paper_bgcolor="white")
    col_g, col_s = st.columns([1, 2])
    with col_g:
        st.plotly_chart(gauge, use_container_width=True)
    with col_s:
        st.markdown('<div class="section-header">Quick Stats</div>', unsafe_allow_html=True)
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Total", f"{total_customers:,}")
        q2.metric("Churned", f"{total_churned:,}")
        q3.metric("Retained", f"{total_customers - total_churned:,}")
        q4.metric("Churn Rate", f"{overall_churn:.2f}%", f"+{overall_churn-20:.1f}% vs target")
        st.markdown('<div class="section-header" style="margin-top:0.5rem">Avg Charges — Churned vs Retained</div>',
                    unsafe_allow_html=True)
        if "Yes" in charges_summary.index and "No" in charges_summary.index:
            m1, m2, m3 = st.columns(3)
            m1.metric("Monthly (Churned)",  f"${charges_summary.loc['Yes','MonthlyCharges']:.2f}",
                      f"${charges_summary.loc['Yes','MonthlyCharges']-charges_summary.loc['No','MonthlyCharges']:.2f} vs Retained")
            m2.metric("Monthly (Retained)", f"${charges_summary.loc['No','MonthlyCharges']:.2f}")
            m3.metric("Avg Tenure (Retained)", f"{charges_summary.loc['No','tenure']:.1f} mo",
                      f"+{charges_summary.loc['No','tenure']-charges_summary.loc['Yes','tenure']:.1f} mo")

    # ── Group tables ──────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Summary Tables</div>', unsafe_allow_html=True)
    tb1, tb2, tb3, tb4 = st.tabs(["Contract", "Internet Service", "Payment Method", "Tenure Group"])

    def style_churn_table(df_t):
        return df_t.style.format({"Churn Rate (%)": "{:.2f}%", "Total": "{:,}", "Churned": "{:,}"})\
                         .background_gradient(subset=["Churn Rate (%)"], cmap="RdYlGn_r", vmin=0, vmax=50)\
                         .set_properties(**{"font-size": "0.88rem"})

    with tb1:
        st.dataframe(style_churn_table(churn_contract), use_container_width=True)
    with tb2:
        st.dataframe(style_churn_table(churn_internet), use_container_width=True)
    with tb3:
        st.dataframe(style_churn_table(churn_payment), use_container_width=True)
    with tb4:
        ten_df = churn_tenure.reset_index()
        ten_df.columns = ["Tenure Group", "Churn Rate (%)"]
        st.dataframe(ten_df.style.format({"Churn Rate (%)": "{:.2f}%"})
                                 .background_gradient(subset=["Churn Rate (%)"], cmap="RdYlGn_r", vmin=0, vmax=55),
                     use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — CHARTS  (Step 4: Visualizations)
# ══════════════════════════════════════════════════════════════
with t4:
    st.markdown('<div class="page-title">📈 Comparative Charts</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">8 interactive charts comparing churn across all key dimensions.</div>',
                unsafe_allow_html=True)

    # ── Row 1: Pie + Contract bar ─────────────────────────────
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown('<div class="section-header">Chart 1 — Overall Churn Distribution</div>',
                    unsafe_allow_html=True)
        pie_data = df["Churn"].value_counts().reset_index()
        pie_data.columns = ["Status", "Count"]
        fig_pie = px.pie(pie_data, names="Status", values="Count",
                         color="Status",
                         color_discrete_map={"No": C_BLUE, "Yes": C_RED},
                         hole=0.42, height=340)
        fig_pie.update_traces(textinfo="percent+label", textfont_size=14,
                               pull=[0, 0.05])
        fig_pie.update_layout(margin=dict(t=10, b=10), showlegend=True,
                               legend=dict(orientation="h", yanchor="bottom", y=-0.08))
        st.plotly_chart(fig_pie, use_container_width=True)

    with r1c2:
        st.markdown('<div class="section-header">Chart 2 — Churn Rate by Contract Type</div>',
                    unsafe_allow_html=True)
        contract_order = ["Month-to-month", "One year", "Two year"]
        c2_df = churn_contract.loc[
            [c for c in contract_order if c in churn_contract.index]
        ].reset_index()
        c2_df.columns = ["Contract", "Total", "Churned", "Churn Rate (%)"]
        fig_c2 = px.bar(c2_df, x="Contract", y="Churn Rate (%)",
                        text="Churn Rate (%)",
                        color="Churn Rate (%)",
                        color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
                        height=340)
        fig_c2.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                              textfont_size=13)
        fig_c2.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=40),
                              yaxis_title="Churn Rate (%)",
                              yaxis=dict(ticksuffix="%"))
        st.plotly_chart(fig_c2, use_container_width=True)

    # ── Row 2: Internet + Payment ─────────────────────────────
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown('<div class="section-header">Chart 3 — Churn Rate by Internet Service</div>',
                    unsafe_allow_html=True)
        i3_df = churn_internet.reset_index()
        i3_df.columns = ["Internet Service", "Total", "Churned", "Churn Rate (%)"]
        i3_df = i3_df.sort_values("Churn Rate (%)", ascending=True)
        fig_c3 = px.bar(i3_df, y="Internet Service", x="Churn Rate (%)",
                        orientation="h", text="Churn Rate (%)",
                        color="Churn Rate (%)",
                        color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
                        height=280)
        fig_c3.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                              textfont_size=13)
        fig_c3.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=20),
                              xaxis=dict(ticksuffix="%"))
        st.plotly_chart(fig_c3, use_container_width=True)

    with r2c2:
        st.markdown('<div class="section-header">Chart 5 — Churn Rate by Payment Method</div>',
                    unsafe_allow_html=True)
        p5_df = churn_payment.reset_index()
        p5_df.columns = ["Payment Method", "Total", "Churned", "Churn Rate (%)"]
        p5_df = p5_df.sort_values("Churn Rate (%)", ascending=True)
        fig_c5 = px.bar(p5_df, y="Payment Method", x="Churn Rate (%)",
                        orientation="h", text="Churn Rate (%)",
                        color="Churn Rate (%)",
                        color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
                        height=280)
        fig_c5.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                              textfont_size=13)
        fig_c5.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=20),
                              xaxis=dict(ticksuffix="%"))
        st.plotly_chart(fig_c5, use_container_width=True)

    # ── Row 3: KDE + Tenure ───────────────────────────────────
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        st.markdown('<div class="section-header">Chart 4 — Monthly Charges: Churned vs Retained</div>',
                    unsafe_allow_html=True)
        fig_c4 = go.Figure()
        for label, colour in [("Yes", C_RED), ("No", C_BLUE)]:
            subset = df[df["Churn"] == label]["MonthlyCharges"]
            fig_c4.add_trace(go.Violin(
                x=df[df["Churn"] == label]["Churn"],
                y=subset,
                name="Churned" if label == "Yes" else "Retained",
                box_visible=True, meanline_visible=True,
                fillcolor=colour, opacity=0.65,
                line_color=colour,
            ))
        fig_c4.update_layout(violinmode="overlay", height=300,
                              yaxis_title="Monthly Charges ($)",
                              xaxis_title="",
                              margin=dict(t=10, b=20),
                              legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_c4, use_container_width=True)

    with r3c2:
        st.markdown('<div class="section-header">Chart 6 — Churn Rate by Tenure Group</div>',
                    unsafe_allow_html=True)
        ten_chart = churn_tenure.reset_index()
        ten_chart.columns = ["Tenure Group", "Churn Rate (%)"]
        fig_c6 = px.bar(ten_chart, x="Tenure Group", y="Churn Rate (%)",
                        text="Churn Rate (%)",
                        color="Churn Rate (%)",
                        color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
                        height=300)
        fig_c6.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                              textfont_size=13)
        fig_c6.update_layout(coloraxis_showscale=False,
                              yaxis=dict(ticksuffix="%"),
                              margin=dict(t=10, b=20))
        st.plotly_chart(fig_c6, use_container_width=True)

    # ── Row 4: Heatmap + Senior grouped bar ──────────────────
    r4c1, r4c2 = st.columns(2)

    with r4c1:
        st.markdown('<div class="section-header">Chart 7 — Churn Rate by Add-on Services (Heatmap)</div>',
                    unsafe_allow_html=True)
        heat_z = heatmap_df.values.tolist()
        heat_x = heatmap_df.columns.tolist()
        heat_y = heatmap_df.index.tolist()
        fig_c7 = go.Figure(go.Heatmap(
            z=heat_z, x=heat_x, y=heat_y,
            colorscale="RdYlGn_r",
            zmin=0, zmax=55,
            text=[[f"{v:.1f}%" for v in row] for row in heat_z],
            texttemplate="%{text}",
            textfont={"size": 13, "color": "white"},
            colorbar=dict(title="Churn %", thickness=14),
        ))
        fig_c7.update_layout(height=320, margin=dict(t=10, b=20),
                              xaxis_title="Subscribed to Service",
                              yaxis_title="")
        st.plotly_chart(fig_c7, use_container_width=True)

    with r4c2:
        st.markdown('<div class="section-header">Chart 8 — Churn Rate: Senior Status × Contract</div>',
                    unsafe_allow_html=True)
        senior_contract_df = (
            df.groupby(["SeniorCitizenLabel", "Contract"])["ChurnFlag"]
            .mean().mul(100).round(2)
            .reset_index()
        )
        senior_contract_df.columns = ["Senior Status", "Contract", "Churn Rate (%)"]
        fig_c8 = px.bar(senior_contract_df, x="Senior Status", y="Churn Rate (%)",
                        color="Contract", barmode="group",
                        text="Churn Rate (%)",
                        color_discrete_sequence=[C_RED, C_AMBER, C_GREEN],
                        height=320)
        fig_c8.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                              textfont_size=11)
        fig_c8.update_layout(margin=dict(t=10, b=20),
                              yaxis=dict(ticksuffix="%"),
                              legend=dict(title="Contract", orientation="h",
                                          yanchor="bottom", y=-0.28))
        st.plotly_chart(fig_c8, use_container_width=True)

    # ── Row 5: Scatter — tenure vs charges (coloured by churn) ─
    st.markdown('<div class="section-header">Chart 9 — Tenure vs Monthly Charges (coloured by Churn)</div>',
                unsafe_allow_html=True)
    fig_scatter = px.scatter(
        df.sample(min(2000, len(df)), random_state=42),
        x="tenure", y="MonthlyCharges",
        color="Churn",
        color_discrete_map={"No": C_BLUE, "Yes": C_RED},
        opacity=0.55, height=360,
        labels={"tenure": "Tenure (months)", "MonthlyCharges": "Monthly Charges ($)"},
        hover_data=["Contract", "InternetService", "PaymentMethod"],
    )
    fig_scatter.update_layout(margin=dict(t=10, b=20),
                               legend=dict(title="Churn", orientation="h",
                                           yanchor="bottom", y=-0.18))
    st.plotly_chart(fig_scatter, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 5 — BUSINESS INSIGHTS  (Step 5: Decisions)
# ══════════════════════════════════════════════════════════════
with t5:
    st.markdown('<div class="page-title">💡 Business Insights & Recommendations</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Data-backed recommendations to reduce churn and improve retention.</div>',
                unsafe_allow_html=True)

    # ── Risk summary cards ────────────────────────────────────
    st.markdown('<div class="section-header">🚨 High-Risk Segments at a Glance</div>',
                unsafe_allow_html=True)
    rs1, rs2, rs3, rs4 = st.columns(4)
    risk_metrics = [
        (rs1, "Month-to-Month Churn",
         f"{churn_contract.loc['Month-to-month','Churn Rate (%)']:.1f}%" if "Month-to-month" in churn_contract.index else "—",
         "vs 2.83% two-year", C_RED),
        (rs2, "Fiber Optic Churn",
         f"{churn_internet.loc['Fiber optic','Churn Rate (%)']:.1f}%" if "Fiber optic" in churn_internet.index else "—",
         "vs 7.40% no internet", C_RED),
        (rs3, "Electronic Check Churn",
         f"{churn_payment.loc['Electronic check','Churn Rate (%)']:.1f}%" if "Electronic check" in churn_payment.index else "—",
         "vs 15.24% auto credit card", C_AMBER),
        (rs4, "First-Year Churn",
         f"{churn_tenure.iloc[0]:.1f}%" if len(churn_tenure) > 0 else "—",
         "critical onboarding window", C_AMBER),
    ]
    for col, label, val, sub, colour in risk_metrics:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="color:{colour}">{val}</div>
              <div class="kpi-delta" style="color:#57606a;font-size:0.75rem">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Recommendations ───────────────────────────────────────
    st.markdown('<div class="section-header">📌 7 Actionable Recommendations</div>',
                unsafe_allow_html=True)

    recs = [
        ("01", "Incentivise Contract Upgrades",
         "Month-to-month customers churn at 42.71% — over 15× higher than two-year contract holders.",
         "🎯 Offer a 10–15% price-locked discount for upgrading to an annual plan at months 3–5 of tenure, when first impressions are formed."),
        ("02", "Audit Fiber Optic Service Quality",
         "Fiber optic subscribers churn at 41.89%, far exceeding DSL (18.96%) customers.",
         "🔧 Conduct a structured satisfaction survey for fiber customers, benchmark pricing vs competitors, and introduce 6-month loyalty discounts."),
        ("03", "Promote Auto-Payment Enrollment",
         "Electronic check payers churn at 45.29% — nearly 3× the rate of credit card auto-pay users (15.24%).",
         "💳 Offer a monthly bill credit ($5–$10) for switching to bank transfer or credit card auto-payment."),
        ("04", "Launch 90-Day Onboarding Programme",
         "Churn rate in the first 12 months is ~47.7% — the most critical window.",
         "🚀 Implement proactive support calls at day 7, 30, and 90; include a welcome discount and service health check."),
        ("05", "Bundle Free Add-On Services at Signup",
         "Customers without OnlineSecurity, TechSupport, or OnlineBackup churn at ~2× the rate of subscribers.",
         "🎁 Offer a 3-month free trial of one add-on (security/backup) at signup to create stickiness and increase perceived value."),
        ("06", "Create a Senior Citizen Service Tier",
         "Senior citizens churn at 41.68% vs 23.61% for non-seniors — nearly double the rate.",
         "👴 Develop simplified plans with dedicated support, larger text/simplified UI, and community partnerships to serve this underserved segment."),
        ("07", "Introduce Price-Lock Guarantees for High-Spend Customers",
         "Churned customers paid an average of $74.44/month vs $61.27 for retained — high-spend customers are more price-sensitive.",
         "🔒 Offer a 12-month price-lock guarantee for customers spending >$70/month to prevent competitor-driven exits."),
    ]

    for num, head, finding, action in recs:
        st.markdown(f"""
        <div class="rec-card">
          <span class="rec-num">#{num}</span>&nbsp;
          <span class="rec-head">{head}</span>
          <div class="rec-body">📊 <strong>Finding:</strong> {finding}</div>
          <div class="rec-action">{action}</div>
        </div>""", unsafe_allow_html=True)

    # ── Priority matrix (impact vs ease) ─────────────────────
    st.markdown('<div class="section-header">🗺 Priority Matrix — Impact vs Implementation Ease</div>',
                unsafe_allow_html=True)
    matrix_df = pd.DataFrame({
        "Recommendation": ["Contract Upgrades", "Auto-Payment Push", "Onboarding Programme",
                            "Add-On Bundles", "Fiber Quality Audit", "Senior Tier", "Price-Lock"],
        "Estimated Impact": [9, 8, 8, 7, 7, 6, 7],
        "Implementation Ease": [8, 9, 7, 9, 5, 4, 6],
        "Priority": ["High", "High", "High", "High", "Medium", "Medium", "Medium"],
    })
    fig_matrix = px.scatter(
        matrix_df, x="Implementation Ease", y="Estimated Impact",
        text="Recommendation", color="Priority",
        color_discrete_map={"High": C_RED, "Medium": C_AMBER},
        size=[60]*7,
        height=380,
    )
    fig_matrix.update_traces(textposition="top center", textfont_size=11)
    fig_matrix.update_layout(
        xaxis=dict(title="Implementation Ease (1=Hard, 10=Easy)", range=[2, 11]),
        yaxis=dict(title="Business Impact (1=Low, 10=High)", range=[4, 11]),
        margin=dict(t=20, b=20),
        legend=dict(title="Priority", orientation="h", yanchor="bottom", y=-0.22),
    )
    fig_matrix.add_vline(x=6.5, line_dash="dot", line_color="#adb5bd")
    fig_matrix.add_hline(y=6.5, line_dash="dot", line_color="#adb5bd")
    st.plotly_chart(fig_matrix, use_container_width=True)
    st.caption("Top-right quadrant = high impact AND easy to implement → prioritize first.")


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  📡 Telecom Customer Churn Analysis &nbsp;|&nbsp;
  Dataset: IBM Telco Customer Churn &nbsp;|&nbsp;
  Built with Python · Streamlit · Plotly
</div>
""", unsafe_allow_html=True)
