import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import BaseEstimator, RegressorMixin


# ============================================================
# NON-NEGATIVE REGRESSION WRAPPER
# ============================================================

class NonNegativeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y):
        if self.model is not None:
            self.model.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if self.model is None:
            return np.zeros(len(X))
        predictions = self.model.predict(X)
        return np.maximum(predictions, 0)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Marketing Intelligence & Campaign Optimization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# UNIFIED CRIMSON/ROSE DARK THEME CSS
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* 1. Global Page Background & Crimson Theme Typography */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: linear-gradient(135deg, #120307 0%, #290812 50%, #170308 100%) !important;
    color: #ffffff !important;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* 2. Global High-Contrast Typography */
h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: #ffffff;
}

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown p {
    color: #ffffff !important;
}

[data-testid="stWidgetLabel"] label, 
[data-testid="stWidgetLabel"] span {
    color: #fecdd3 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* 3. Sidebar Dark Crimson Theme */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0205 0%, #21050e 50%, #330714 100%) !important;
    border-right: 1px solid #4c111e !important;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

.sidebar-title {
    font-size: 1.35rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
    color: #ffffff !important;
}

.sidebar-subtitle {
    color: #fca5a5 !important;
    font-size: 0.88rem;
}

/* 4. Banner Headers */
.main-header {
    padding: 1.8rem 2.2rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #881337 0%, #be123c 50%, #e11d48 100%);
    box-shadow: 0 10px 25px rgba(225, 29, 72, 0.35);
    border: 1px solid #f43f5e;
    margin-bottom: 1.5rem;
}

.main-header h1 {
    color: #ffffff !important;
    margin: 0 0 0.4rem 0;
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}

.main-header p {
    color: #ffe4e6 !important;
    margin: 0;
    font-size: 1rem;
}

.section-banner {
    padding: 0.75rem 1.25rem;
    border-radius: 12px;
    background: linear-gradient(90deg, #9f1239, #e11d48);
    box-shadow: 0 4px 12px rgba(159, 18, 57, 0.35);
    margin: 1rem 0 1.25rem 0;
}

.section-banner h2 {
    color: #ffffff !important;
    font-size: 1.2rem;
    font-weight: 700;
    margin: 0;
}

/* 5. Custom Card Components (Cohesive Dark Theme) */
.info-card {
    padding: 1.25rem 1.5rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #280710 0%, #1e040b 100%);
    border: 1px solid #e11d48;
    border-left: 5px solid #f43f5e;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
    margin-bottom: 1rem;
}

.info-card h3 {
    color: #ffffff !important;
    margin-top: 0;
    font-size: 1.15rem;
    font-weight: 700;
}

.info-card p {
    color: #ffe4e6 !important;
    margin-bottom: 0;
    font-size: 0.95rem;
    line-height: 1.55;
}

.model-card {
    padding: 1.2rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #24060e 0%, #1a0308 100%);
    border: 1px solid #4c111e;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3);
    min-height: 130px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.model-card:hover {
    border-color: #f43f5e;
    transform: translateY(-2px);
}

.model-icon {
    font-size: 1.5rem;
}

.model-title {
    color: #ffffff !important;
    font-size: 1rem;
    font-weight: 700;
    margin-top: 0.3rem;
}

.model-description {
    color: #fecdd3 !important;
    font-size: 0.85rem;
    margin-top: 0.2rem;
}

/* 6. Native Streamlit Metric Cards (Cohesive Dark Theme) */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #26060f 0%, #1d0309 100%) !important;
    border: 1px solid #e11d48 !important;
    border-radius: 14px !important;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3) !important;
}

[data-testid="stMetricLabel"] * {
    color: #fecdd3 !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
}

[data-testid="stMetricValue"] * {
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 1.75rem !important;
}

[data-testid="stMetricDelta"] * {
    font-weight: 700 !important;
}

/* 7. Input Fields & Dropdowns */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
input {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #fda4af !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

div[data-baseweb="select"] * {
    color: #0f172a !important;
}

/* Dropdown Options Popover Menu Fix */
[data-baseweb="popover"],
[data-baseweb="menu"],
div[role="listbox"],
ul[role="listbox"] {
    background-color: #ffffff !important;
    border: 1px solid #fda4af !important;
    border-radius: 8px !important;
}

[data-baseweb="popover"] li,
[data-baseweb="popover"] li *,
[data-baseweb="menu"] li,
[data-baseweb="menu"] li *,
div[role="option"],
div[role="option"] * {
    background-color: #ffffff !important;
    color: #0f172a !important;
    font-weight: 600 !important;
}

[data-baseweb="popover"] li:hover,
[data-baseweb="popover"] li:hover *,
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] li:hover *,
div[role="option"]:hover,
div[role="option"]:hover * {
    background-color: #ffe4e6 !important;
    color: #be123c !important;
}

/* 8. Tabs Styling Fixes */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent !important;
    padding-bottom: 6px;
}

.stTabs [data-baseweb="tab"] {
    height: auto !important;
    padding: 8px 18px !important;
    background-color: #2e0912 !important;
    border: 1px solid #5c1325 !important;
    border-radius: 8px !important;
    color: #fecdd3 !important;
    font-weight: 700 !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #ffffff !important;
    border-color: #f43f5e !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #be123c, #e11d48) !important;
    border-color: #f43f5e !important;
}

.stTabs [aria-selected="true"] * {
    color: #ffffff !important;
}

/* 9. Native Alert Fixes (Pure High-Contrast White Text) */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    background-color: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid #f43f5e !important;
}

div[data-testid="stAlert"] * {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* 10. Crimson Action Button */
.stButton > button {
    border-radius: 10px;
    border: none;
    background: linear-gradient(90deg, #e11d48, #f43f5e);
    color: #ffffff !important;
    font-weight: 700;
    font-size: 1.05rem;
    padding: 0.75rem 1.4rem;
    box-shadow: 0 4px 15px rgba(225, 29, 72, 0.4);
}

.stButton > button:hover {
    background: linear-gradient(90deg, #be123c, #e11d48);
    color: #ffffff !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown(
    """
<div class="main-header">
    <h1>📊 Marketing Intelligence & Campaign Optimization</h1>
    <p>AI-powered campaign performance prediction, financial forecasting, and profitability analysis.</p>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR NAVIGATION & INFO
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">📊 Marketing Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sidebar-subtitle">Campaign Decision Support System</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### About")
    st.write(
        """
        This decision support system leverages machine learning models to
        evaluate digital advertising configurations and forecast expected ROI,
        CTR, conversions, and net profitability prior to budget commitment.
        """
    )

    st.markdown("---")
    st.markdown("### Core Predictive Engine")
    st.markdown(
        """
        🎯 **CTR Prediction Engine**
        
        🔄 **Conversion Rate Forecasting**
        
        💰 **Revenue Optimization Model**
        
        📈 **Profitability Classifier**
        """
    )

    st.markdown("---")
    st.markdown("### Technical Architecture")
    st.markdown(
        """
        • **Model:** HistGradientBoosting
        • **Inputs:** 15 Campaign Features
        • **Processed Dimensions:** 49 Features
        • **Decision Rule:** Multi-objective threshold
        """
    )

    st.markdown("---")
    st.caption("Marketing Intelligence Project • v2.0")


# ============================================================
# LOAD SAVED MODELS WITH SIMULATION FALLBACK
# ============================================================

@st.cache_resource
def load_models():
    """Loads trained scikit-learn pipeline models from the models directory."""
    paths = {
        "ctr": "models/ctr_best_model.pkl",
        "conversion_rate": "models/conversion_rate_best_model.pkl",
        "revenue": "models/revenue_best_model.pkl",
        "profitability": "models/profitability_best_model.pkl",
    }

    loaded_models = {}
    missing_files = []

    for key, path in paths.items():
        if os.path.exists(path):
            try:
                loaded_models[key] = joblib.load(path)
            except Exception:
                missing_files.append(path)
        else:
            missing_files.append(path)

    if missing_files:
        return None, missing_files

    return (
        loaded_models["ctr"],
        loaded_models["conversion_rate"],
        loaded_models["revenue"],
        loaded_models["profitability"],
    ), []


models, missing_model_files = load_models()
models_loaded = models is not None

if models_loaded:
    ctr_model, conversion_rate_model, revenue_model, profitability_model = models
else:
    ctr_model = conversion_rate_model = revenue_model = profitability_model = None


# ============================================================
# DEMO / SIMULATION PREDICTION ENGINE
# ============================================================

def generate_heuristic_predictions(input_df: pd.DataFrame):
    """Fallback engine calculating realistic predictions when pkl models are absent."""
    row = input_df.iloc[0]
    spend = float(row["ad_spend"])
    quality = float(row["quality_score"])
    cta_bonus = 0.25 if row["has_call_to_action"] == 1 else 0.0
    retarget_bonus = 0.35 if row["retargeting_flag"] == 1 else 0.0

    ctr = max(0.5, min(12.0, (quality * 0.35) + cta_bonus + retarget_bonus + np.random.uniform(0.8, 1.5)))
    conv_rate = max(0.2, min(8.5, (quality * 0.25) + (retarget_bonus * 1.5) + np.random.uniform(0.5, 1.2)))
    roi_multiplier = (quality / 10.0) * 1.4 + (1.2 if row["retargeting_flag"] == 1 else 0.8)
    estimated_rev = spend * roi_multiplier

    prob_profit = max(0.05, min(0.98, (estimated_rev - spend) / spend * 0.5 + 0.5))
    is_profitable = 1 if prob_profit >= 0.5 else 0

    return ctr, conv_rate, estimated_rev, prob_profit, is_profitable


# ============================================================
# NAVIGATION TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🏠 Overview",
        "🎯 Campaign Configuration",
        "📈 Predictions",
        "💡 Recommendation",
    ]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:
    st.markdown(
        '<div class="section-banner"><h2>🚀 Intelligent Campaign Decision System</h2></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="info-card">
    <h3>What does this application do?</h3>
    <p>
        The Marketing Intelligence System leverages trained machine learning algorithms to evaluate campaign parameters, 
        forecast click-through rates (CTR), conversion efficiency, estimated total revenue, and profitability metrics prior to ad spend commitment.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
<div class="model-card">
    <div class="model-icon">🎯</div>
    <div class="model-title">CTR Prediction</div>
    <div class="model-description">Forecast expected click-through rate across ad networks.</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
<div class="model-card">
    <div class="model-icon">🔄</div>
    <div class="model-title">Conversion Rate</div>
    <div class="model-description">Estimate expected visitor-to-customer conversion efficiency.</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
<div class="model-card">
    <div class="model-icon">💰</div>
    <div class="model-title">Revenue Forecast</div>
    <div class="model-description">Predict gross financial revenue generated by the campaign.</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
<div class="model-card">
    <div class="model-icon">📈</div>
    <div class="model-title">Profitability</div>
    <div class="model-description">Classify campaign profitability and return on ad spend.</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.write("")

    st.markdown(
        """
<div class="info-card">
    <h3>🔄 System Workflow</h3>
    <p>
        <b>1. Campaign Setup:</b> Input target parameters, ad copy attributes, and spend allocation.<br>
        <b>2. Multi-Model Inference:</b> Ensemble estimators evaluate engagement and financial returns.<br>
        <b>3. Financial Analytics:</b> Instant breakdown of projected Net Profit, ROAS, and Conversion metrics.<br>
        <b>4. Strategic Recommendation:</b> Automated risk-adjusted decision guidance for marketing leads.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    if models_loaded:
        st.success("✅ All four machine learning models are active and loaded successfully.")
    else:
        st.info("ℹ️ Running in Simulation Mode. Place valid `.pkl` models in `models/` directory for live production inference.")


# ============================================================
# TAB 2 — CAMPAIGN CONFIGURATION
# ============================================================

with tab2:
    st.markdown(
        '<div class="section-banner"><h2>🎯 Configure Your Campaign</h2></div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 1. Campaign Characteristics & Audience Targeting")

    col1, col2, col3 = st.columns(3)

    with col1:
        campaign_objective = st.selectbox(
            "Campaign Objective",
            ["Sales", "Brand Awareness", "Traffic"],
            key="campaign_objective",
        )

        platform = st.selectbox(
            "Platform",
            ["Google", "Facebook", "Instagram", "Twitter", "LinkedIn"],
            key="platform",
        )

        device_type = st.selectbox(
            "Device Type",
            ["Mobile", "Desktop", "Tablet"],
            key="device_type",
        )

    with col2:
        creative_format = st.selectbox(
            "Creative Format",
            ["Image", "Video", "Carousel"],
            key="creative_format",
        )

        ad_copy_length = st.selectbox(
            "Ad Copy Length",
            ["Short", "Medium", "Long"],
            key="ad_copy_length",
        )

        income_bracket = st.selectbox(
            "Income Bracket",
            ["Low", "Medium", "High"],
            key="income_bracket",
        )

    with col3:
        purchase_intent_score = st.selectbox(
            "Purchase Intent Score",
            ["Low", "Medium", "High"],
            key="purchase_intent_score",
        )

        day_of_week = st.selectbox(
            "Day of Week",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
            key="day_of_week",
        )

        industry_vertical = st.selectbox(
            "Industry Vertical",
            [
                "Technology",
                "Retail",
                "Finance",
                "Healthcare",
                "Travel",
                "Education",
                "Automotive",
            ],
            key="industry_vertical",
        )

    st.markdown("### 2. Operational Parameters & Budget")

    col1, col2, col3 = st.columns(3)

    with col1:
        has_call_to_action = st.selectbox(
            "Call to Action Included",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
            key="has_call_to_action",
        )

        retargeting_flag = st.selectbox(
            "Retargeting Campaign",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
            key="retargeting_flag",
        )

    with col2:
        quarter = st.selectbox("Quarter", [1, 2, 3, 4], key="quarter")

        campaign_day = st.number_input(
            "Campaign Duration (Days)",
            min_value=1,
            max_value=365,
            value=15,
            step=1,
            key="campaign_day",
        )

    with col3:
        quality_score = st.number_input(
            "Ad Quality Score (1-10)",
            min_value=1,
            max_value=10,
            value=8,
            step=1,
            key="quality_score",
        )

        ad_spend = st.number_input(
            "Planned Ad Spend (₹)",
            min_value=0.0,
            value=10000.0,
            step=1000.0,
            key="ad_spend",
        )

    campaign_input = pd.DataFrame(
        [
            {
                "campaign_objective": campaign_objective,
                "platform": platform,
                "device_type": device_type,
                "creative_format": creative_format,
                "ad_copy_length": ad_copy_length,
                "income_bracket": income_bracket,
                "purchase_intent_score": purchase_intent_score,
                "day_of_week": day_of_week,
                "industry_vertical": industry_vertical,
                "has_call_to_action": has_call_to_action,
                "retargeting_flag": retargeting_flag,
                "quarter": quarter,
                "campaign_day": campaign_day,
                "quality_score": quality_score,
                "ad_spend": ad_spend,
            }
        ]
    )

    st.divider()

    predict_button = st.button(
        "🚀 Predict Campaign Performance",
        type="primary",
        use_container_width=True,
    )

    if predict_button:
        if ad_spend <= 0:
            st.warning("⚠️ Please specify a planned ad spend greater than ₹0.")
        else:
            with st.spinner("Processing campaign inputs..."):
                if models_loaded:
                    predicted_ctr = float(ctr_model.predict(campaign_input)[0])
                    predicted_conversion_rate = float(
                        conversion_rate_model.predict(campaign_input)[0]
                    )
                    predicted_revenue = float(
                        revenue_model.predict(campaign_input)[0]
                    )
                    profitability_probability = float(
                        profitability_model.predict_proba(campaign_input)[0, 1]
                    )
                    profitability_prediction = int(
                        profitability_model.predict(campaign_input)[0]
                    )
                else:
                    (
                        predicted_ctr,
                        predicted_conversion_rate,
                        predicted_revenue,
                        profitability_probability,
                        profitability_prediction,
                    ) = generate_heuristic_predictions(campaign_input)

                estimated_profit = predicted_revenue - ad_spend
                estimated_roas = (
                    (predicted_revenue / ad_spend) if ad_spend > 0 else 0.0
                )

                if profitability_probability >= 0.70 and estimated_profit > 0:
                    recommendation = "HIGH POTENTIAL"
                elif profitability_probability >= 0.40 and estimated_profit > 0:
                    recommendation = "MODERATE POTENTIAL"
                else:
                    recommendation = "LOW POTENTIAL"

                st.session_state["prediction_results"] = {
                    "ctr": predicted_ctr,
                    "conversion_rate": predicted_conversion_rate,
                    "revenue": predicted_revenue,
                    "profitability_probability": profitability_probability,
                    "profitability_prediction": profitability_prediction,
                    "profit": estimated_profit,
                    "roas": estimated_roas,
                    "ad_spend": ad_spend,
                    "recommendation": recommendation,
                }

            st.success("✅ Prediction completed! Switch to the **📈 Predictions** tab to view analytics.")


# ============================================================
# TAB 3 — PREDICTIONS & ANALYTICS
# ============================================================

with tab3:
    st.markdown(
        '<div class="section-banner"><h2>📈 Campaign Predictions & Analytics</h2></div>',
        unsafe_allow_html=True,
    )

    if "prediction_results" not in st.session_state:
        st.info("👉 Please configure your campaign in the **🎯 Campaign Configuration** tab and click **Predict Campaign Performance**.")
    else:
        results = st.session_state["prediction_results"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Predicted CTR",
                value=f"{results['ctr']:.2f}%",
            )

        with col2:
            st.metric(
                label="Conversion Rate",
                value=f"{results['conversion_rate']:.2f}%",
            )

        with col3:
            st.metric(
                label="Projected Revenue",
                value=f"₹{results['revenue']:,.2f}",
            )

        with col4:
            st.metric(
                label="Profitability Confidence",
                value=f"{results['profitability_probability'] * 100:.1f}%",
            )

        st.write("")
        st.markdown("### 💰 Financial Performance Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Planned Ad Spend",
                value=f"₹{results['ad_spend']:,.2f}",
            )

        with col2:
            st.metric(
                label="Estimated Profit",
                value=f"₹{results['profit']:,.2f}",
                delta=f"{((results['profit'] / results['ad_spend']) * 100):.1f}% Margin" if results['ad_spend'] > 0 else None,
            )

        with col3:
            st.metric(
                label="Estimated ROAS",
                value=f"{results['roas']:.2f}x",
            )

        st.write("")

        # ----------------------------------------------------
        # HIGH VISUAL IMPACT PLOTLY CHARTS
        # ----------------------------------------------------
        col_chart1, col_chart2 = st.columns([1.1, 1])

        with col_chart1:
            # Multi-colored Bar Chart
            fin_metrics = ["Ad Spend", "Projected Revenue", "Net Profit"]
            fin_amounts = [
                results["ad_spend"],
                results["revenue"],
                results["profit"],
            ]
            
            # Distinct vibrant color palette for each financial metric
            bar_colors = ["#f43f5e", "#06b6d4", "#10b981"]

            fig_fin = go.Figure(
                data=[
                    go.Bar(
                        x=fin_metrics,
                        y=fin_amounts,
                        text=[f"₹{val:,.2f}" for val in fin_amounts],
                        textposition="outside",
                        marker=dict(
                            color=bar_colors,
                            line=dict(color="#ffffff", width=1.5),
                            pattern_shape="",
                        ),
                        hovertemplate="<b>%{x}</b><br>Amount: ₹%{y:,.2f}<extra></extra>",
                    )
                ]
            )

            fig_fin.update_layout(
                title=dict(
                    text="<b>Financial Breakdown Comparison</b>",
                    font=dict(color="#ffffff", size=17, family="Plus Jakarta Sans, sans-serif"),
                    x=0,
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ffffff", family="Plus Jakarta Sans, sans-serif", size=13),
                xaxis=dict(
                    title="",
                    tickfont=dict(color="#ffffff", size=13, weight="bold"),
                    showgrid=False,
                ),
                yaxis=dict(
                    title="Amount (₹)",
                    title_font=dict(color="#fecdd3", size=13),
                    tickfont=dict(color="#fecdd3", size=12),
                    gridcolor="rgba(244, 63, 94, 0.15)",
                    zerolinecolor="rgba(244, 63, 94, 0.3)",
                ),
                margin=dict(l=20, r=20, t=50, b=20),
                height=360,
            )
            fig_fin.update_traces(
                textfont=dict(color="#ffffff", size=12, family="Plus Jakarta Sans, sans-serif", weight="bold"),
            )
            st.plotly_chart(fig_fin, use_container_width=True)

        with col_chart2:
            # Vibrant Gradient Multi-Tier Gauge Chart
            gauge_val = results["profitability_probability"] * 100

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=gauge_val,
                    domain={"x": [0, 1], "y": [0.15, 1]},
                    title={
                        "text": "<b>Profitability Confidence Score</b>",
                        "font": {
                            "color": "#ffffff",
                            "size": 17,
                            "family": "Plus Jakarta Sans, sans-serif",
                        },
                    },
                    number={
                        "suffix": "%",
                        "valueformat": ".1f",
                        "font": {
                            "color": "#38bdf8",
                            "size": 38,
                            "family": "Plus Jakarta Sans, sans-serif",
                            "weight": "bold",
                        },
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickcolor": "#ffffff",
                            "tickfont": {
                                "color": "#fecdd3",
                                "size": 11,
                                "family": "Plus Jakarta Sans, sans-serif",
                            },
                            "tickwidth": 2,
                            "ticklen": 6,
                        },
                        "bar": {"color": "#38bdf8", "thickness": 0.28},
                        "bgcolor": "#1e040b",
                        "borderwidth": 2,
                        "bordercolor": "#e11d48",
                        "steps": [
                            {"range": [0, 40], "color": "#be123c"},
                            {"range": [40, 70], "color": "#d97706"},
                            {"range": [70, 100], "color": "#059669"},
                        ],
                        "threshold": {
                            "line": {"color": "#ffffff", "width": 4},
                            "thickness": 0.85,
                            "value": 70,
                        },
                    },
                )
            )
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ffffff", family="Plus Jakarta Sans, sans-serif"),
                margin=dict(l=30, r=30, t=50, b=10),
                height=360,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.write("")

        # ----------------------------------------------------
        # SECONDARY VISUALIZATION ROW: REVENUE & COST DISTRIBUTION
        # ----------------------------------------------------
        st.markdown("### 📊 Budget Allocation & Revenue Efficiency")
        col_pie1, col_pie2 = st.columns(2)

        with col_pie1:
            # Donut chart showing Cost vs Net Profit distribution
            pie_labels = ["Planned Spend", "Estimated Net Profit"]
            pie_values = [results["ad_spend"], max(0, results["profit"])]
            pie_colors = ["#f43f5e", "#10b981"]

            fig_donut = go.Figure(
                data=[
                    go.Pie(
                        labels=pie_labels,
                        values=pie_values,
                        hole=0.55,
                        marker=dict(colors=pie_colors, line=dict(color="#1e040b", width=3)),
                        hoverinfo="label+value+percent",
                        textinfo="percent+label",
                        textfont=dict(color="#ffffff", size=13, weight="bold"),
                    )
                ]
            )
            fig_donut.update_layout(
                title=dict(
                    text="<b>Revenue Share: Cost vs Net Profit</b>",
                    font=dict(color="#ffffff", size=16, family="Plus Jakarta Sans, sans-serif"),
                    x=0,
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(
                    font=dict(color="#ffffff", size=12),
                    orientation="h",
                    y=-0.1,
                ),
                margin=dict(l=20, r=20, t=50, b=30),
                height=300,
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_pie2:
            # Performance Multiplier Comparison Chart
            metrics_labels = ["Ad Quality", "CTR (%)", "Conv. Rate (%)", "ROAS (x)"]
            metrics_values = [
                st.session_state.get("quality_score", 8),
                results["ctr"],
                results["conversion_rate"],
                results["roas"],
            ]
            mult_colors = ["#a855f7", "#ec4899", "#3b82f6", "#10b981"]

            fig_radar = go.Figure(
                data=[
                    go.Bar(
                        x=metrics_labels,
                        y=metrics_values,
                        text=[f"{v:.2f}" for v in metrics_values],
                        textposition="auto",
                        marker=dict(color=mult_colors, line=dict(color="#ffffff", width=1)),
                    )
                ]
            )
            fig_radar.update_layout(
                title=dict(
                    text="<b>Core Campaign Efficiency Metrics</b>",
                    font=dict(color="#ffffff", size=16, family="Plus Jakarta Sans, sans-serif"),
                    x=0,
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(tickfont=dict(color="#ffffff", size=12, weight="bold")),
                yaxis=dict(
                    title="Scale",
                    title_font=dict(color="#fecdd3", size=12),
                    tickfont=dict(color="#fecdd3", size=11),
                    gridcolor="rgba(244, 63, 94, 0.15)",
                ),
                margin=dict(l=20, r=20, t=50, b=20),
                height=300,
            )
            st.plotly_chart(fig_radar, use_container_width=True)


# ============================================================
# TAB 4 — RECOMMENDATION
# ============================================================

with tab4:
    st.markdown(
        '<div class="section-banner"><h2>💡 Business Recommendation & Action Plan</h2></div>',
        unsafe_allow_html=True,
    )

    if "prediction_results" not in st.session_state:
        st.info("👉 Run a campaign prediction first in the **🎯 Campaign Configuration** tab.")
    else:
        results = st.session_state["prediction_results"]
        recommendation = results["recommendation"]

        if recommendation == "HIGH POTENTIAL":
            description = (
                "This campaign demonstrates strong key performance indicators and a robust ROI outlook. "
                "Projected revenue safely exceeds planned spend with high statistical confidence."
            )
        elif recommendation == "MODERATE POTENTIAL":
            description = (
                "This campaign shows moderate growth potential. Positive returns are expected, "
                "but optimizing creative assets or quality scores prior to full budget rollout is recommended."
            )
        else:
            description = (
                "Under current settings, this campaign carries a risk of negative net returns. "
                "We recommend revising targeting parameters or ad spend allocations before launching."
            )

        st.markdown(
            f"""
<div class="info-card">
    <h3 style="color: #ffffff !important; font-size: 1.25rem;">🎯 Status: {recommendation}</h3>
    <p style="color: #ffe4e6 !important; font-size: 0.98rem; line-height: 1.55;">{description}</p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.write("")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Profitability Probability",
                value=f"{results['profitability_probability'] * 100:.1f}%",
            )

        with col2:
            st.metric(
                label="Estimated Net Profit",
                value=f"₹{results['profit']:,.2f}",
            )

        with col3:
            st.metric(
                label="Target ROAS",
                value=f"{results['roas']:.2f}x",
            )

        st.write("")
        st.markdown("### Recommended Business Actions")

        if recommendation == "HIGH POTENTIAL":
            st.success(
                "✅ **Proceed with Campaign Deployment:** The configuration shows strong commercial promise. "
                "Consider scaling budget incrementally while monitoring early performance trends."
            )
        elif recommendation == "MODERATE POTENTIAL":
            st.warning(
                "⚠️ **A/B Testing Recommended:** Conduct small-scale test flights before committing full budget. "
                "Focus on improving Quality Score and incorporating retargeting audience segments."
            )
        else:
            st.error(
                "❌ **Hold Budget & Re-evaluate:** Restructure campaign parameters before launching. "
                "Prioritize optimizing Call-to-Action placement or targeting higher purchase intent audiences."
            )