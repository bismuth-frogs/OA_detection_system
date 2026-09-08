"""
frontend.py -- Streamlit frontend supporting standalone and combined multimodal OA risk screening.
"""

import base64
import io
import json

import requests
import streamlit as st
from PIL import Image

API_URL = "http://localhost:8000"

st.set_page_config(page_title="OA Multimodal Screening", page_icon="🦴", layout="wide")

st.title("🦴 Integrated Osteoarthritis Risk Screening Portal")
st.caption("Combines clinical questionnaire evaluation and knee X-ray imaging into an integrated risk score.")

# --- Backend Health Check Sidebar ---
with st.sidebar:
    st.header("Backend Connection")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        if health.get("status") == "ok":
            st.success("API Connected")
            st.write(f"**Tabular Model:** {'Ready' if health.get('tabular_model_loaded') else 'Offline'}")
            st.write(f"**Vision Model:** {'Ready' if health.get('vision_model_loaded') else 'Offline'}")
        else:
            st.warning("Backend models not fully ready.")
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach API at {API_URL}. Ensure uvicorn is running.")

# --- Navigation Tabs ---
tab_combined, tab_questionnaire, tab_xray = st.tabs([
    "🔗 Combined Assessment (Recommended)",
    "📋 Clinical Questionnaire Only",
    "🩻 X-Ray Analysis Only"
])

# ==============================================================================
# TAB 1: COMBINED ASSESSMENT (X-RAY + QUESTIONNAIRE)
# ==============================================================================
with tab_combined:
    st.subheader("Unified Multimodal Screening")
    st.write("Fill in patient clinical factors and upload a knee X-ray for a comprehensive evaluation.")

    with st.form("combined_form"):
        st.markdown("##### 1. Clinical & Lifestyle Factors")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=100, value=55, key="comb_age")
            bmi = st.number_input("BMI (kg/m²)", min_value=14.0, max_value=55.0, value=28.5, step=0.1, key="comb_bmi")
            sex_str = st.selectbox("Sex", ["Female", "Male"], key="comb_sex")
        with c2:
            injury_str = st.selectbox("History of Joint Injury/Trauma", ["No", "Yes"], key="comb_inj")
            family_str = st.selectbox("Family History of OA", ["No", "Yes"], key="comb_fam")
        with c3:
            strain_str = st.selectbox("Occupational Knee Strain", ["Low", "Moderate", "High / Heavy Lifting"], key="comb_str")
            activity_str = st.selectbox("Physical Activity Level", ["Sedentary", "Moderate", "High / Vigorous"], key="comb_act")

        st.markdown("##### 2. Knee X-Ray Image")
        uploaded_xray = st.file_uploader("Upload Knee X-ray (PNG or JPG)", type=["png", "jpg", "jpeg"], key="comb_img")

        submit_combined = st.form_submit_button("Generate Combined Risk Assessment", type="primary")

    if submit_combined:
        if uploaded_xray is None:
            st.error("Please upload an X-ray image to perform a combined assessment.")
        else:
            clinical_payload = {
                "age": int(age),
                "bmi": float(bmi),
                "sex": 1 if sex_str == "Male" else 0,
                "joint_injury_history": 1 if injury_str == "Yes" else 0,
                "occupational_strain": {"Low": 0, "Moderate": 1, "High / Heavy Lifting": 2}[strain_str],
                "family_history": 1 if family_str == "Yes" else 0,
                "activity_level": {"Sedentary": 0, "Moderate": 1, "High / Vigorous": 2}[activity_str]
            }

            files = {
                "image": (uploaded_xray.name, uploaded_xray.getvalue(), uploaded_xray.type)
            }
            data = {
                "clinical_data": json.dumps(clinical_payload)
            }

            with st.spinner("Processing multimodal fusion pipeline..."):
                try:
                    res = requests.post(f"{API_URL}/predict-combined", data=data, files=files, timeout=30)
                    if res.status_code == 200:
                        out = res.json()
                        st.divider()

                        # --- OVERALL COMBINED RESULT ---
                        comb_cat = out["combined_risk_category"]
                        comb_prob = out["combined_risk_probability"]
                        comb_pct = out["combined_risk_percentage"]

                        st.markdown(f"## 🎯 Combined Screening Result: **{comb_cat}** ({comb_pct})")
                        st.progress(comb_prob)

                        if "High" in comb_cat:
                            st.error("⚠️ High overall risk detected across clinical history and radiographic imaging.")
                        elif "Moderate" in comb_cat:
                            st.warning("⚠️ Moderate risk indicated. Clinical monitoring and preventative measures recommended.")
                        else:
                            st.success("✅ Low overall risk detected based on current inputs.")

                        st.divider()

                        # --- BREAKDOWN COLUMNS ---
                        col_left, col_right = st.columns(2)

                        with col_left:
                            st.markdown("### 📋 Clinical Breakdown")
                            clin = out["clinical_breakdown"]
                            st.write(f"**Clinical Risk Level:** {clin['risk_category']}")
                            st.write(f"**Clinical Risk Score:** {clin['risk_percentage']}")
                            st.progress(clin["risk_probability"])

                        with col_right:
                            st.markdown("### 🩻 Imaging Breakdown (Grad-CAM)")
                            img_data = out["imaging_breakdown"]
                            st.write(f"**Detected Class:** {img_data['prediction']}")
                            st.write(f"**Confidence:** {img_data['confidence']:.1%}")

                            # Render Heatmap
                            b64_data = img_data["heatmap_image"].split(",", 1)[1]
                            heatmap_bytes = base64.b64decode(b64_data)
                            st.image(Image.open(io.BytesIO(heatmap_bytes)), caption="Grad-CAM Heatmap", use_container_width=True)

                    else:
                        st.error(f"API Error ({res.status_code}): {res.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Failed to connect to backend server.")

# ==============================================================================
# TAB 2: QUESTIONNAIRE ONLY
# ==============================================================================
with tab_questionnaire:
    st.subheader("Clinical Risk Questionnaire")
    with st.form("solo_clinical_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            q_age = st.number_input("Age", min_value=18, max_value=100, value=50)
            q_bmi = st.number_input("BMI", min_value=14.0, max_value=55.0, value=25.0)
            q_sex = st.selectbox("Sex", ["Female", "Male"])
        with c2:
            q_inj = st.selectbox("Joint Injury", ["No", "Yes"])
            q_fam = st.selectbox("Family History", ["No", "Yes"])
        with c3:
            q_str = st.selectbox("Occupational Strain", ["Low", "Moderate", "High / Heavy Lifting"])
            q_act = st.selectbox("Activity Level", ["Sedentary", "Moderate", "High / Vigorous"])
        q_sub = st.form_submit_button("Calculate Questionnaire Risk")

    if q_sub:
        payload = {
            "age": int(q_age), "bmi": float(q_bmi), "sex": 1 if q_sex == "Male" else 0,
            "joint_injury_history": 1 if q_inj == "Yes" else 0,
            "occupational_strain": {"Low": 0, "Moderate": 1, "High / Heavy Lifting": 2}[q_str],
            "family_history": 1 if q_fam == "Yes" else 0,
            "activity_level": {"Sedentary": 0, "Moderate": 1, "High / Vigorous": 2}[q_act]
        }
        res = requests.post(f"{API_URL}/predict-tabular", json=payload).json()
        st.markdown(f"### Predicted Risk: **{res['risk_category']}** ({res['risk_percentage']})")
        st.progress(res["risk_probability"])

# ==============================================================================
# TAB 3: X-RAY ONLY
# ==============================================================================
# ==============================================================================
# TAB 3: X-RAY ONLY
# ==============================================================================
with tab_xray:
    st.subheader("Radiographic X-Ray Assessment")
    solo_file = st.file_uploader("Upload Knee X-ray", type=["png", "jpg", "jpeg"], key="solo_xray")
    if solo_file and st.button("Analyze Image"):
        with st.spinner("Analyzing X-ray image..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict", 
                    files={"image": (solo_file.name, solo_file.getvalue(), solo_file.type)}
                )
                if response.status_code == 200:
                    res = response.json()
                    st.write(f"**Prediction:** {res['prediction']} ({res['confidence']:.1%})")
                    
                    if "heatmap_image" in res:
                        b64 = res["heatmap_image"].split(",", 1)[1]
                        st.image(Image.open(io.BytesIO(base64.b64decode(b64))), caption="Grad-CAM Overlay")
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend API server.")
