import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
import shap
import joblib
import agent

st.set_page_config(page_title="Executive Dashboard", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

ALL_FEATURES = ['IS_DIRECTION_2', 'LOCKDOWN_L4_AUCKLAND', 'LOCKDOWN_L3_AUCKLAND', 'LOCKDOWN_L4_NATIONAL', 'LOCKDOWN_L3_NATIONAL', 'LOCKDOWN_L2_NATIONAL', 'EVENT_Lockdown_Announced', 'IS_WEEKEND', 'HOLIDAY_ANZAC_Day', 'HOLIDAY_Christmas_Day', 'HOLIDAY_New_Years_Day', 'HOLIDAY_Boxing_Day', 'HOLIDAY_Waitangi_Day', 'HOLIDAY_Day_after_New_Years_Day', 'HOLIDAY_Queens_Birthday', 'HOLIDAY_Labour_Day', 'HOLIDAY_Easter_Monday', 'HOLIDAY_Good_Friday', 'EVENT_Multi Hazard, Snow / Ice', 'EVENT_High Wind / Gust', 'EVENT_Multi Hazard', 'EVENT_Snow / Ice', 'EVENT_Flooding', 'IS_AUCKLAND', 'IS_WELLINGTON', 'IS_CANTERBURY', 'HOUR_SIN', 'HOUR_COS', 'DAY_SIN', 'DAY_COS', 'MONTH_SIN', 'MONTH_COS', 'YEAR', 'TEMP', 'RH', 'WDSP', 'DEWP', 'VISIB', 'GUST', 'FLOW_lag_24h']

NRC_FEATURES = ['LOCKDOWN_L4_AUCKLAND', 'LOCKDOWN_L3_AUCKLAND', 'LOCKDOWN_L4_NATIONAL', 'LOCKDOWN_L3_NATIONAL', 'LOCKDOWN_L2_NATIONAL', 'EVENT_Lockdown_Announced', 'IS_WEEKEND', 'HOLIDAY_ANZAC_Day', 'HOLIDAY_Christmas_Day', 'HOLIDAY_New_Years_Day', 'HOLIDAY_Boxing_Day', 'HOLIDAY_Waitangi_Day', 'HOLIDAY_Day_after_New_Years_Day', 'HOLIDAY_Queens_Birthday', 'HOLIDAY_Labour_Day', 'HOLIDAY_Easter_Monday', 'HOLIDAY_Good_Friday', 'EVENT_Multi Hazard, Snow / Ice', 'EVENT_High Wind / Gust', 'EVENT_Multi Hazard', 'EVENT_Snow / Ice', 'EVENT_Flooding', 'TEMP', 'RH', 'WDSP', 'DEWP', 'VISIB', 'GUST']

@st.cache_data
def load_predictions():
    try:
        # Load the pre-computed predictions CSV
        df = pd.read_csv('predictions.csv', dtype={'SITEREF': str})
        df['DATETIME_HOUR'] = pd.to_datetime(df['DATETIME_HOUR'])
        return df
    except Exception as e:
        return None

predictions_df = load_predictions()

@st.cache_resource
def load_model():
    try:
        return joblib.load('tuned_xgb(final).joblib')
    except Exception as e:
        return None

xgb_model = load_model()

# --- PREDICTION FUNCTION ---
def get_prediction_row(date, selected_time, direction, siteref):
    if predictions_df is None:
        return None
        
    target_dt = pd.to_datetime(f"{date} {selected_time.strftime('%H:%M:%S')}")
    is_dir_2 = 1 if direction == "Incoming" else 0
    
    match = predictions_df[
        (predictions_df['SITEREF'].astype(str) == str(siteref)) &
        (predictions_df['DATETIME_HOUR'] == target_dt) &
        (predictions_df['IS_DIRECTION_2'] == is_dir_2)
    ]
    return match if not match.empty else None

def predict_traffic(region, date, selected_time, direction, siteref):
    match = get_prediction_row(date, selected_time, direction, siteref)
    if match is not None:
        if 'Predicted_Flow' in match.columns:
            return int(match['Predicted_Flow'].iloc[0])
        elif 'XGBoost_Pred' in match.columns:
            return int(match['XGBoost_Pred'].iloc[0])
        else:
            return int(match.iloc[0, -1]) 
    else:
        return None

def get_daily_predictions(date, direction, siteref):
    if predictions_df is None:
        return None
        
    is_dir_2 = 1 if direction == "Incoming" else 0
    date_str = str(date)
    
    match = predictions_df[
        (predictions_df['SITEREF'].astype(str) == str(siteref)) &
        (predictions_df['DATETIME_HOUR'].dt.date.astype(str) == date_str) &
        (predictions_df['IS_DIRECTION_2'] == is_dir_2)
    ].sort_values('DATETIME_HOUR')
    
    return match

def create_shap_waterfall(model, row_df):
    import plotly.graph_objects as go
    if model is None or row_df is None:
        return None
        
    # Check if we have the features
    missing_features = [f for f in ALL_FEATURES if f not in row_df.columns]
    if missing_features:
        st.error(f"Missing features in data for SHAP computation: {missing_features}")
        return None

    X = row_df[ALL_FEATURES]
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)[0]
    base_value = explainer.expected_value
    
    if isinstance(base_value, np.ndarray):
        base_value = base_value[0]
        
    feature_names = ALL_FEATURES
    feature_impacts = [(feature_names[i], shap_vals[i]) for i in range(len(feature_names))]
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    
    top_impacts = feature_impacts[:15]
    other_impacts = sum([x[1] for x in feature_impacts[15:]])
    
    y_cats = []
    x_vals = []
    text = []
    colors = []
    
    for name, impact in top_impacts:
        display_name = name.replace("EVENT_", "").replace("HOLIDAY_", "").replace("_", " ")
        if name in NRC_FEATURES:
            display_name = f"🚨 {display_name}"
            
        y_cats.append(display_name)
        x_vals.append(impact)
        sign = "+" if impact > 0 else ""
        text.append(f"{sign}{impact:,.0f} vph")
        colors.append("#d32f2f" if impact > 0 else "#38a169")
        
    y_cats.append("Other Features")
    x_vals.append(other_impacts)
    sign = "+" if other_impacts > 0 else ""
    text.append(f"{sign}{other_impacts:,.0f} vph")
    colors.append("#d32f2f" if other_impacts > 0 else "#38a169")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="SHAP", orientation="h",
        y=y_cats,
        x=x_vals,
        textposition="outside",
        text=text,
        marker_color=colors,
        showlegend=False
    ))
    
    # Dummy traces for legend
    fig.add_trace(go.Bar(name="Increases Traffic", x=[None], y=[None], marker_color="#d32f2f"))
    fig.add_trace(go.Bar(name="Decreases Traffic", x=[None], y=[None], marker_color="#38a169"))
    
    fig.update_layout(
        title=dict(text="Feature Impact on Prediction (SHAP Values)", font=dict(family="Inter", size=14, color="#4a5568")),
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=60, t=50, b=50),
        yaxis=dict(autorange="reversed"),  # To show largest impact at top
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5, font=dict(size=12))
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f0f2f6')
    fig.update_yaxes(showgrid=False)
    
    return fig

def create_nrc_factors_table(model, row_df):
    if model is None or row_df is None:
        return None
    X = row_df[ALL_FEATURES]
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)[0]
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value[0]
        
    nrc_data = []
    for i, feature in enumerate(ALL_FEATURES):
        if feature in NRC_FEATURES:
            val = row_df[feature].iloc[0]
            impact = shap_vals[i]
            if abs(impact) > 5: # Show if it has a meaningful impact (> 5 vph)
                display_name = feature.replace("EVENT_", "").replace("HOLIDAY_", "").replace("_", " ")
                
                # Format Current Value properly for weather features vs boolean events
                if feature in ['TEMP', 'DEWP']: current_val_str = f"{val:.1f} °C"
                elif feature == 'RH': current_val_str = f"{val:.1f} %"
                elif feature in ['WDSP', 'GUST']: current_val_str = f"{val:.1f} km/h"
                elif feature == 'VISIB': current_val_str = f"{val:.1f} km"
                else: current_val_str = "Active" if val != 0 else "Inactive"
                
                nrc_data.append({
                    "Feature": display_name,
                    "Current Value": current_val_str,
                    "SHAP Impact": impact
                })
    
    if not nrc_data:
        return pd.DataFrame([{"Feature": "No active NRCs", "Current Value": "-", "SHAP Impact": "-"}])
        
    df = pd.DataFrame(nrc_data)
    df = df.sort_values(by="SHAP Impact", key=abs, ascending=False)
    df['SHAP Impact'] = df['SHAP Impact'].apply(lambda x: f"+{(x/base_value)*100:.1f}%" if x > 0 else f"{(x/base_value)*100:.1f}%")
    return df

def get_lag_interaction_table(model, row_df):
    if model is None or row_df is None:
        return None
    X = row_df[ALL_FEATURES]
    explainer = shap.TreeExplainer(model)
    # Get interaction values for this specific row (shape: 1 x Features x Features)
    interact_vals = explainer.shap_interaction_values(X)[0]
    
    if "FLOW_lag_24h" not in ALL_FEATURES:
        return None
    lag_idx = ALL_FEATURES.index("FLOW_lag_24h")
    
    interaction_data = []
    main_effect = interact_vals[lag_idx, lag_idx]
    
    for i, feature in enumerate(ALL_FEATURES):
        if i == lag_idx:
            continue
        impact = interact_vals[lag_idx, i]
        # Only include if absolute impact is greater than 10 vph
        if abs(impact) > 10:
            display_name = feature.replace("EVENT_", "").replace("HOLIDAY_", "").replace("_", " ")
            if feature in NRC_FEATURES:
                display_name = f"🚨 {display_name}"
            
            interaction_data.append({
                "Interacting Feature": display_name,
                "Current Value": row_df[feature].iloc[0],
                "Lag Impact Modification": impact
            })
            
    if not interaction_data:
        df = pd.DataFrame([{"Interacting Feature": "No major interactions", "Current Value": "-", "Lag Impact Modification": "-"}])
        return df, main_effect
        
    df = pd.DataFrame(interaction_data)
    df = df.sort_values(by="Lag Impact Modification", key=abs, ascending=False)
    
    # Format modification as raw vph
    df['Lag Impact Modification'] = df['Lag Impact Modification'].apply(
        lambda x: f"+{x:,.0f} vph" if isinstance(x, (int, float)) and x > 0 else (f"{x:,.0f} vph" if isinstance(x, (int, float)) else x)
    )
    
    return df, main_effect

def create_prediction_chart(daily_df, selected_time, scenario="Current"):
    import plotly.graph_objects as go
    
    if daily_df.empty:
        return None
    
    pred_col = 'Predicted_Flow' if 'Predicted_Flow' in daily_df.columns else 'XGBoost_Pred'
    if pred_col not in daily_df.columns:
        pred_col = daily_df.columns[-1] 
        
    baseline_col = 'FLOW_lag_168h' if 'FLOW_lag_168h' in daily_df.columns else ('FLOW_lag_24h' if 'FLOW_lag_24h' in daily_df.columns else 'FLOW')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_df['DATETIME_HOUR'],
        y=daily_df[baseline_col],
        name='Historical Baseline',
        line=dict(color='#8fa0ba', width=2, dash='dash'),
        hovertemplate='Time: %{x|%H:%M}<br>Baseline: %{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=daily_df['DATETIME_HOUR'],
        y=daily_df[pred_col],
        name='ML Prediction',
        line=dict(color='#2b6cb0', width=3),
        hovertemplate='Time: %{x|%H:%M}<br>Predicted: %{y:,.0f}<extra></extra>'
    ))
    
    target_dt = pd.to_datetime(f"{daily_df['DATETIME_HOUR'].dt.date.iloc[0]} {selected_time.strftime('%H:%M:%S')}")
    end_dt = target_dt + pd.Timedelta(hours=4)
    future_df = daily_df[(daily_df['DATETIME_HOUR'] >= target_dt) & (daily_df['DATETIME_HOUR'] <= end_dt)]
    
    peak_time = None
    if not future_df.empty:
        peak_row = future_df.loc[future_df[pred_col].idxmax()]
        peak_time = peak_row['DATETIME_HOUR']
        
        highlight_start = peak_time - pd.Timedelta(minutes=30)
        highlight_end = peak_time + pd.Timedelta(minutes=30)
        
        fig.add_vrect(
            x0=highlight_start, x1=highlight_end,
            fillcolor="#ffe0e0", opacity=0.5,
            layer="below", line_width=0,
            annotation_text="Predicted Congestion Event", 
            annotation_position="top left",
            annotation_font_color="#e53e3e"
        )
        
        fig.add_trace(go.Scatter(
            x=[peak_time], y=[peak_row[pred_col]],
            mode='markers',
            marker=dict(color='black', size=8, line=dict(color='#2b6cb0', width=2)),
            showlegend=False,
            hoverinfo='skip'
        ))
        
    if scenario == "Future":
        fig.update_xaxes(range=[target_dt - pd.Timedelta(hours=1), end_dt + pd.Timedelta(hours=1)])
            
    fig.update_layout(
        title=dict(text='Volume Prediction Timeline', font=dict(family='Playfair Display', size=22, color='#1c2b4d')),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=60, b=0)
    )
    
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='#f0f2f6',
        tickformat='%H:%M', dtick=14400000,
        showline=True, linewidth=1, linecolor='#eef1f5', color='#8fa0ba'
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f2f6', showline=False, color='#8fa0ba', tickformat=',')
    
    return fig

# --- UI SETUP ---

# Inject Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@700&display=swap');

/* Hide Streamlit components */
#MainMenu {visibility: hidden;}
/* header {visibility: hidden;} */
footer {visibility: hidden;}

/* Global font */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f8f9fa;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: white;
    border-right: 1px solid #eef1f5;
}

/* Header bar */
.top-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 2rem;
    background-color: white;
    border-bottom: 1px solid #eef1f5;
    margin-top: -4rem; 
    margin-bottom: 2rem;
    margin-left: -4rem;
    margin-right: -4rem;
}
.logo-area {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-weight: 700;
    font-size: 1.5rem;
    letter-spacing: 1px;
    color: #1c2b4d;
}
.logo-icon {
    color: #2b6cb0;
    font-size: 1.8rem;
}
.top-icons {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    color: #4a5568;
}
.profile-icon {
    background-color: #e2e8f0;
    border-radius: 50%;
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 0.8rem;
    color: #1a202c;
}

/* Dashboard specific styles */
.dashboard-title {
    color: #1c2b4d;
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    font-size: 2.2rem;
    margin-bottom: 0.2rem;
}
.dashboard-subtitle {
    color: #8fa0ba;
    font-size: 1rem;
    margin-bottom: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.export-btn {
    border: 1px solid #eef1f5;
    padding: 0.4rem 1rem;
    border-radius: 4px;
    font-size: 0.85rem;
    color: #4a5568;
    background: white;
    cursor: pointer;
    font-family: 'Inter', sans-serif;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

.summary-card {
    background-color: white;
    border-radius: 4px;
    padding: 2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    border: 1px solid #eef1f5;
    margin-bottom: 2.5rem;
}
.summary-title {
    color: #1c2b4d;
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.summary-text {
    color: #4a5568;
    line-height: 1.8;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}
.action-buttons {
    display: flex;
    gap: 1rem;
}
.action-btn {
    border: 1px dashed #4b89f5;
    color: #4b89f5;
    background: transparent;
    padding: 0.4rem 0.8rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 0.2rem;
    position: relative;
    letter-spacing: 0.5px;
}
.connected-badge {
    position: absolute;
    top: -10px;
    left: 0;
    background-color: #4b89f5;
    color: white;
    font-size: 0.5rem;
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
}

.section-title {
    color: #1c2b4d;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 1rem;
}

.metrics-row {
    display: flex;
    gap: 1.5rem;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: white;
    border-radius: 4px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    border: 1px solid #eef1f5;
    flex: 1;
    display: flex;
    flex-direction: column;
}
.metric-card.anomaly {
    border-left: 4px solid #e53e3e;
}
.metric-header {
    display: flex;
    justify-content: space-between;
    color: #8fa0ba;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
    letter-spacing: 0.5px;
}
.metric-value {
    color: #1c2b4d;
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: auto;
    line-height: 1;
}
.metric-value.highlight {
    color: #2b6cb0;
}
.metric-value-sub {
    font-size: 1rem;
    color: #8fa0ba;
    font-family: 'Inter', sans-serif;
    font-weight: 400;
}
.anomaly-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #1c2b4d;
    line-height: 1.3;
    margin-bottom: auto;
}
.metric-footer {
    margin-top: 1.5rem;
    font-size: 0.85rem;
    color: #8fa0ba;
    display: flex;
    align-items: center;
}
.metric-delta {
    display: inline-flex;
    align-items: center;
    color: #2b6cb0;
    background-color: #ebf8ff;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.5rem;
}
.anomaly-impact {
    background-color: #f8f9fa;
    border: 1px solid #eef1f5;
    padding: 0.3rem 0.6rem;
    border-radius: 4px;
    font-size: 0.75rem;
    color: #4a5568;
    font-weight: 600;
}

/* Specific styling overrides for Streamlit radio buttons */
[data-testid="stRadio"] label {
    font-family: 'Playfair Display', serif;
    color: #1c2b4d;
    padding: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Top Header (Logo and profile)
st.markdown("""
<div class="top-header">
    <div class="logo-area">
        <span class="logo-icon">🔷</span> Traffic Xplained
    </div>
    <div class="top-icons">
        <span>🔔</span>
        <span>⚙️</span>
        <div class="profile-icon">JD</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR: USER INPUTS ---
st.sidebar.markdown("<h3 style='color:#8fa0ba; font-size:0.75rem; font-weight:700; letter-spacing:1px; margin-bottom:1rem;'>REGIONAL SELECTION</h3>", unsafe_allow_html=True)

def get_site_count(region_name):
    if predictions_df is not None and 'REGION_NAME' in predictions_df.columns:
        return predictions_df[predictions_df['REGION_NAME'] == region_name]['SITEREF'].nunique()
    # fallback counts based on dictionary below
    mapping = {"Auckland": 5, "Wellington": 5, "Canterbury": 5}
    return mapping.get(region_name, 0)

regions = ["Auckland", "Wellington", "Canterbury"]
selected_region = st.sidebar.radio(
    "Region", 
    regions, 
    label_visibility="collapsed",
    format_func=lambda x: f"{x} — {get_site_count(x)} sites"
)

st.sidebar.markdown("<br><hr style='border-top: 1px solid #eef1f5; margin: 1rem 0;'><br>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='color:#8fa0ba; font-size:0.75rem; font-weight:700; letter-spacing:1px; margin-bottom:1rem;'>PREDICTION PARAMETERS</h3>", unsafe_allow_html=True)

# Inputs
selected_date = st.sidebar.date_input(
    "Date", 
    value=datetime(2021, 5, 11).date(),
    min_value=datetime(2021, 1, 1).date(),
    max_value=datetime(2021, 12, 31).date()
)
hours = [time(i, 0) for i in range(24)]
# Defaulting to 15:00
default_time_idx = 15
selected_time = st.sidebar.selectbox("Time", hours, index=default_time_idx, format_func=lambda x: x.strftime('%H:%M'))

directions = ["Incoming", "Outgoing"]
selected_direction = st.sidebar.selectbox("Direction", directions)

if predictions_df is not None and 'REGION_NAME' in predictions_df.columns:
    region_siterefs = sorted(predictions_df[predictions_df['REGION_NAME'] == selected_region]['SITEREF'].unique().tolist())
else:
    # fallback mappings
    region_mapping = {
        "Auckland": ["01600024", "01600058", "01610003", "01610008", "01610009"],
        "Wellington": ["01N01058", "01N01064", "01N01076", "01N01077", "01N01078"],
        "Canterbury": ["01S00348", "01S10341", "01S10342", "01S20334", "01S20337"]
    }
    region_siterefs = region_mapping.get(selected_region, ["01600024"])

meta_df = agent.load_meta_context()
def format_siteref(siteref):
    if meta_df is not None:
        match = meta_df[meta_df['SITEREF'] == str(siteref)]
        if not match.empty:
            row = match.iloc[0]
            region_clean = str(row['REGION']).split(' - ')[-1] if ' - ' in str(row['REGION']) else str(row['REGION'])
            # e.g., "01600024 (SH 16, Auckland)"
            return f"{siteref} (SH {row['SH']}, {region_clean})"
    return siteref

selected_siteref = st.sidebar.selectbox("Site Reference (Siteref)", region_siterefs, format_func=format_siteref)

predict_button = st.sidebar.button("Generate Dashboard", type="primary", use_container_width=True)

if 'dashboard_visible' not in st.session_state:
    st.session_state.dashboard_visible = False

if predict_button:
    st.session_state.dashboard_visible = True

# --- MAIN CONTENT AREA ---

if st.session_state.dashboard_visible:
    # Run prediction
    predicted_volume = predict_traffic(selected_region, selected_date, selected_time, selected_direction, selected_siteref)
    
    if predicted_volume is None:
        pred_text = "N/A"
    else:
        pred_text = f"{predicted_volume:,}"
    
    # Calculate peak time dynamically for the selected date
    daily_df_peak = get_daily_predictions(selected_date, selected_direction, selected_siteref)
    if daily_df_peak is not None and not daily_df_peak.empty:
        p_col = 'Predicted_Flow' if 'Predicted_Flow' in daily_df_peak.columns else 'XGBoost_Pred'
        if p_col not in daily_df_peak.columns:
            p_col = daily_df_peak.columns[-1]
        peak_row = daily_df_peak.loc[daily_df_peak[p_col].idxmax()]
        peak_time = peak_row['DATETIME_HOUR'].strftime('%H:%M')
    else:
        peak_time = "N/A"
    
    html_content = f"""
<div class="dashboard-title">Executive Dashboard</div>
<div class="dashboard-subtitle">
    <span>{selected_region} Region Overview</span>
    <button class="export-btn">📥 Export Report</button>
</div>


"""
    st.markdown(html_content, unsafe_allow_html=True)

    if 'show_shap' not in st.session_state:
        st.session_state.show_shap = False

    # Action buttons using Streamlit columns
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        st.button("VIEW 24-HOUR FORECAST ➔", use_container_width=True)
    with col_b:
        if st.button("ANALYZE DRIVERS 🔍", type="primary", use_container_width=True):
            st.session_state.show_shap = not st.session_state.show_shap

    if st.session_state.show_shap:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-title' style='font-family: \"Playfair Display\", serif; font-size: 1.8rem;'><span style='color: #2b6cb0;'> </span> Why This Prediction?</div>", unsafe_allow_html=True)
        row_df = get_prediction_row(selected_date, selected_time, selected_direction, selected_siteref)
        if row_df is not None and xgb_model is not None:
            tab1, tab2, tab3 = st.tabs(["Visual Analysis", "Detailed Breakdown", "Agentic Explanation"])
            
            with tab1:
                col_shap1, col_shap2 = st.columns([3, 2])
                with col_shap1:
                    fig_shap = create_shap_waterfall(xgb_model, row_df)
                    if fig_shap:
                        st.plotly_chart(fig_shap, use_container_width=True)
                with col_shap2:
                    st.markdown("<h4 style='font-family: \"Playfair Display\", serif; color: #1c2b4d;'>Top Contributing NRC Factors</h4>", unsafe_allow_html=True)
                    nrc_df = create_nrc_factors_table(xgb_model, row_df)
                    if nrc_df is not None:
                        st.dataframe(nrc_df, use_container_width=True, hide_index=True)
                    
                    with st.expander("Why did Historical Lag impact change?"):
                        res = get_lag_interaction_table(xgb_model, row_df)
                        if res:
                            interact_df, main_effect = res
                            sign = "+" if main_effect > 0 else ""
                            st.markdown(f"**Pure mathematical effect of Lag 24h:** `{sign}{main_effect:,.0f} vph`")
                            st.markdown("<span style='font-size:0.85rem; color:#4a5568;'>How other conditions actively modified the lag's reliability:</span>", unsafe_allow_html=True)
                            st.dataframe(interact_df, use_container_width=True, hide_index=True)
            
            with tab2:
                # Add the 4 breakdown boxes below
                explainer = shap.TreeExplainer(xgb_model)
                shap_vals_raw = explainer.shap_values(row_df[ALL_FEATURES])[0]
                
                def get_impact(feat_list):
                    return sum([shap_vals_raw[ALL_FEATURES.index(f)] for f in feat_list if f in ALL_FEATURES])
                    
                time_impact = get_impact(['HOUR_SIN', 'HOUR_COS'])
                day_impact = get_impact(['DAY_SIN', 'DAY_COS', 'IS_WEEKEND'])
                lag_impact = get_impact(['FLOW_lag_24h'])
                weather_impact = get_impact(['TEMP', 'RH', 'WDSP', 'DEWP', 'VISIB', 'GUST'])
                
                time_val = selected_time.strftime("%H:%M") + (" (Peak)" if 15 <= selected_time.hour <= 18 else "")
                day_val = selected_date.strftime("%A") + (" PM" if selected_time.hour >= 12 else " AM")
                lag_str = "Above Avg" if lag_impact > 0 else "Below Avg"
                weather_str = "Normal" if abs(weather_impact) < 20 else ("Active Weather")
                
                def format_box(title, val, impact):
                    color = "#e53e3e" if impact > 0 else "#38a169"
                    sign = "+" if impact > 0 else ""
                    icon = "📈" if impact > 0 else "📉"
                    return f"""
<div style="flex: 1; background: white; padding: 1.2rem; border-radius: 6px; border: 1px solid #eef1f5; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
    <div style="font-size: 0.85rem; color: #8fa0ba; margin-bottom: 0.3rem;">{title}</div>
    <div style="font-size: 1.2rem; font-weight: 700; color: #1c2b4d; margin-bottom: 0.5rem;">{val}</div>
    <div style="font-size: 0.95rem; color: {color}; font-weight: 600;">
        {icon} {sign}{int(impact)} vph
    </div>
</div>
"""
                    
                boxes_html = f"""
<div style="display: flex; gap: 1rem; margin-top: 1rem; margin-bottom: 1rem;">
    {format_box("Time of Day", time_val, time_impact)}
    {format_box("Day Pattern", day_val, day_impact)}
    {format_box("Historical Lag", lag_str, lag_impact)}
    {format_box("Weather", weather_str, weather_impact)}
</div>
"""
                st.markdown(boxes_html, unsafe_allow_html=True)
                
            with tab3:
                st.markdown("<h4 style='color: #1c2b4d; font-family: \"Playfair Display\", serif;'>Agentic Traffic Analysis</h4>", unsafe_allow_html=True)
                
                baseline_val = row_df['FLOW_lag_24h'].iloc[0] if 'FLOW_lag_24h' in row_df.columns and row_df['FLOW_lag_24h'].iloc[0] > 0 else 1000
                
                # Add a generate button so we don't spam the API unnecessarily
                if st.button("🪄 Generate Agentic breakdown", type="primary", key="btn_agent_gen", use_container_width=True):
                    
                    # Extract top SHAP factors into a readable string
                    explainer = shap.TreeExplainer(xgb_model)
                    shap_vals_arr = explainer.shap_values(row_df[ALL_FEATURES])[0]
                    
                    feature_impacts = [(ALL_FEATURES[i], shap_vals_arr[i]) for i in range(len(ALL_FEATURES))]
                    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
                    top_impacts = feature_impacts[:15]  # Top 10 factors
                    
                    shap_str = ""
                    for name, impact in top_impacts:
                        if abs(impact) > 5: # Only include meaningful impacts
                            sign = "+" if impact > 0 else ""
                            display_name = name.replace("EVENT_", "Alert: ").replace("HOLIDAY_", "Holiday: ").replace("_", " ")
                            
                            # Append actual feature value for weather so LLM has context
                            val = row_df[name].iloc[0]
                            if name in ['TEMP', 'DEWP']: val_str = f" ({val:.1f} °C)"
                            elif name == 'RH': val_str = f" ({val:.1f} %)"
                            elif name in ['WDSP', 'GUST']: val_str = f" ({val:.1f} km/h)"
                            elif name == 'VISIB': val_str = f" ({val:.1f} km)"
                            else: val_str = ""
                            
                            shap_str += f"- **{display_name}**{val_str}: {sign}{impact:,.0f} vph\n"
                            
                    # Inject specific extreme event name if available
                    if 'EXTREME_ID' in row_df.columns:
                        ext_id = row_df['EXTREME_ID'].iloc[0]
                        if pd.notna(ext_id) and ext_id != "None" and str(ext_id).strip() != "":
                            clean_ext_id = str(ext_id).replace("_", " ")
                            shap_str += f"\n**CRITICAL EVENT CONTEXT**: The generic SHAP feature for the active weather/hazard alert actually represents the '{clean_ext_id}'. You MUST rename the generic SHAP driver (e.g., 'Multi Hazard') to this specific event name in the table. Do NOT create a separate or extra row for this context.\n"
                    
                    st.markdown("<hr>", unsafe_allow_html=True)
                    
                    with st.spinner("Agent is analyzing traffic patterns and state highway context..."):
                        # Call agent stream
                        response_stream = agent.generate_traffic_explanation(
                            siteref=selected_siteref,
                            date_str=selected_date.strftime('%A, %b %d, %Y'),
                            time_str=selected_time.strftime('%I:%M %p'),
                            direction=selected_direction,
                            predicted_volume=predicted_volume,
                            baseline_volume=baseline_val,
                            top_shap_factors_str=shap_str
                        )
                        
                        # Use Streamlit's stream writer
                        st.write_stream(response_stream)
                
        else:
            st.error("Model or prediction row not available to calculate SHAP.")
            
        st.markdown("<hr>", unsafe_allow_html=True)

    html_content2 = f"""
<div class="section-title">Current Network Status</div>

<div class="metrics-row">
<div class="metric-card">
<div class="metric-header"><span>PREDICTED TRAFFIC</span> <span>🚘</span></div>
<div class="metric-value">{pred_text} <span class="metric-value-sub">vph</span></div>
<div class="metric-footer"><span class="metric-delta">↗ +8.4%</span> vs. historical avg</div>
</div>

<div class="metric-card">
<div class="metric-header"><span>EXPECTED PEAK TIME</span> <span>⏱️</span></div>
<div class="metric-value highlight">{peak_time}</div>
<div class="metric-footer"><b>Duration:</b>&nbsp;Approx. 2.5 hours</div>
</div>

<div class="metric-card anomaly" style="border-left: 4px solid #2b6cb0;">
<div class="metric-header"><span>ROUTE SELECTION</span> <span style="color:#2b6cb0">🛣️</span></div>
<div class="anomaly-title">Site: {selected_siteref}<br>{selected_direction}</div>
<div class="metric-footer"><span class="anomaly-impact">🚦 Active Monitoring</span></div>
</div>
</div>
"""
    st.markdown(html_content2, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualization section
    daily_df = get_daily_predictions(selected_date, selected_direction, selected_siteref)
    if daily_df is not None and not daily_df.empty:
        st.markdown("<div class='section-title' style='margin-top: 20px;'>Timeline Scenario</div>", unsafe_allow_html=True)
        scenario = st.radio("Timeframe", ["Current", "Future"], horizontal=True, label_visibility="collapsed")
        
        fig = create_prediction_chart(daily_df, selected_time, scenario)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No timeline data available for the selected date.")

else:
    # Empty state
    st.info("Please verify your parameters in the sidebar and click 'Generate Dashboard' to view the analytics.")
