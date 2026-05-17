import os
import pandas as pd
import google.generativeai as genai

# Load the highway metadata
def load_meta_context():
    meta_path = "step_b1_meta.pkl"
    try:
        df = pd.read_pickle(meta_path)
        return df
    except Exception as e:
        return None

def get_site_context(siteref):
    df = load_meta_context()
    if df is None:
        return "Geographic metadata unavailable."
    
    match = df[df['SITEREF'] == str(siteref)]
    if not match.empty:
        row = match.iloc[0]
        region = row['REGION']
        sh = row['SH']
        lane = row['LANE']
        lat = row['LAT']
        lon = row['LON']
        return f"State Highway {sh} in the {region} Region (Lane Configuration: {lane}). Coordinates: {lat}, {lon}."
    return "Site context not found in metadata."

def generate_traffic_explanation(
    siteref, date_str, time_str, direction, 
    predicted_volume, baseline_volume, 
    top_shap_factors_str
):
    # Try reading from Streamlit Secrets first, then environment variables
    api_key = None
    try:
        import streamlit as st
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass
        
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
        
    if not api_key:
        yield "⚠️ **API Key Missing**: Please add your API key to `.streamlit/secrets.toml` or set the `GEMINI_API_KEY` environment variable."
        return
        
    genai.configure(api_key=api_key)

    site_context = get_site_context(siteref)
    
    diff = predicted_volume - baseline_volume
    if baseline_volume > 0:
        diff_pct = (diff / baseline_volume) * 100
    else:
        diff_pct = 0
    higher_lower = "higher" if diff > 0 else "lower"
    
    system_instruction = """You are an expert dual-persona AI:
1. A rigorous Data Scientist explaining machine learning (XGBoost) traffic predictions using SHAP values.
2. A strategic Traffic Management Advisor providing actionable, real-world recommendations to operators.

Your goal is to explain WHY the model predicted a certain traffic volume and WHAT the operator should do about it.

Guidelines:
- Explain the key driving factors derived from the SHAP values (e.g. why the baseline was adjusted up or down). Translate technical terms (like 'FLOW_lag_24h' or 'HOUR_SIN') into plain English (e.g., 'Historical traffic at this time yesterday', 'Time of day effect').
- Provide 2-3 specific, actionable traffic management recommendations based on the conditions (e.g., Variable Message Signs, Signal Phasing, Public Alerts).
- Maintain a professional, analytical, yet urgent tone suitable for a control room environment.

CRITICAL DESIGN & LAYOUT INSTRUCTIONS:
You must format your response using highly scannable, premium Markdown layout. Follow this exact structure:

> Executive Summary : 
> [Provide a 2-sentence bold summary of the prediction vs the baseline here. Make it punchy.]

### 📊 Understanding the Prediction (Why?)
[Provide a brief 1-sentence intro]
| Key Driver | Impact on Traffic | Explanation |
| :--- | :--- | :--- |
| **[Driver Name]** | [e.g., +450 vph] | [Plain English explanation of why this factor matters right now] |
*(Format the SHAP factors as a clean Markdown table like above. Translate technical terms into plain English.)*

---

### 🚦 Actionable Recommendations
[Provide 2-3 specific, actionable traffic management recommendations based on the conditions]
* **[Emoji] [Intervention Name]**: [Detailed action plan]
* **[Emoji] [Intervention Name]**: [Detailed action plan]
"""

    prompt = f"""
### Current Situation
- **Site Reference:** {siteref}
- **Geographic Context:** {site_context}
- **Target Date & Time:** {date_str} at {time_str}
- **Traffic Direction:** {direction}

### Prediction Details
- **Predicted Volume:** {predicted_volume:,.0f} vehicles/hour
- **Historical Baseline (Lag 24h):** {baseline_volume:,.0f} vehicles/hour
- **Deviation:** {abs(diff_pct):.1f}% {higher_lower} than the historical baseline.

### Key Drivers (SHAP Value Impacts)
These are the most significant factors that pushed the prediction up or down from the baseline (in vehicles per hour):
{top_shap_factors_str}

Based on this data, provide the agentic explanation and recommendations.
"""

    try:
        model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=system_instruction)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.3),
            stream=True
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"⚠️ **Error generating explanation**: {str(e)}\n\nMake sure your Gemini API key is valid and you have internet connectivity."
