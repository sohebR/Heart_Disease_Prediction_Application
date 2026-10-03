import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CardioPredict | Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main application */
    .main {
        padding-top: 1.5rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Header */
    .hero {
        padding: 1.5rem 1.8rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        background: linear-gradient(
            135deg,
            rgba(255, 75, 75, 0.16),
            rgba(255, 255, 255, 0.03)
        );
        border: 1px solid rgba(255, 255, 255, 0.10);
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 750;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.72;
        line-height: 1.6;
    }

    /* Section cards */
    .section-card {
        padding: 1.2rem 1.3rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.025);
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 650;
        margin-bottom: 0.15rem;
    }

    .section-description {
        font-size: 0.82rem;
        opacity: 0.60;
        margin-bottom: 1rem;
    }

    /* Result cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 18px;
        margin-top: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.10);
        background: rgba(255, 255, 255, 0.035);
    }

    .result-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.6;
        margin-bottom: 0.4rem;
    }

    .result-title {
        font-size: 1.65rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .result-class {
        font-size: 0.9rem;
        opacity: 0.65;
    }

    /* Metric cards */
    .metric-card {
        padding: 1rem;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
    }

    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
    }

    .metric-label {
        font-size: 0.75rem;
        opacity: 0.6;
        margin-top: 0.2rem;
    }

    /* Disclaimer */
    .disclaimer {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        background: rgba(255, 193, 7, 0.08);
        border: 1px solid rgba(255, 193, 7, 0.22);
        font-size: 0.82rem;
        line-height: 1.55;
        margin-top: 1.5rem;
    }

    /* Sidebar */
    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .sidebar-subtitle {
        font-size: 0.8rem;
        opacity: 0.6;
        line-height: 1.5;
        margin-bottom: 1.2rem;
    }

    /* Button */
    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 3rem;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    package = joblib.load(
        "models/heart_disease_model.joblib"
    )

    return package


model_package = load_model()

model = model_package["model"]
category_maps = model_package["category_maps"]
categorical_features = model_package["categorical_features"]
feature_columns = model_package["feature_columns"]
target_classes = model_package["target_classes"]


# ============================================================
# CLASS LABELS
# ============================================================

class_labels = {
    0: "No Heart Disease",
    1: "Heart Disease - Class 1",
    2: "Heart Disease - Class 2",
    3: "Heart Disease - Class 3",
    4: "Heart Disease - Class 4"
}


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_patient(patient_data):

    patient_df = pd.DataFrame([patient_data])

    # Convert unavailable fields to actual missing values
    patient_df = patient_df.replace(
        "Not provided",
        np.nan
    )

    # Apply the exact categorical mappings
    # used during model training
    for col in categorical_features:

        patient_df[col] = patient_df[col].map(
            category_maps[col]
        )

    # Ensure exact feature order
    patient_df = patient_df[feature_columns]

    prediction = model.predict(
        patient_df
    )[0]

    probabilities = model.predict_proba(
        patient_df
    )[0]

    return prediction, probabilities


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">❤️ CardioPredict</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'Machine-learning based multiclass heart disease '
        'prediction system.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Model Overview")

    st.metric(
        "Input Features",
        len(feature_columns)
    )

    st.metric(
        "Prediction Classes",
        len(target_classes)
    )

    st.metric(
        "Model",
        "HistGradientBoosting"
    )

    st.divider()

    st.markdown("### Prediction Classes")

    for class_id in target_classes:

        st.write(
            f"**Class {class_id}** — "
            f"{class_labels[class_id]}"
        )

    st.divider()

    st.caption(
        "The model was trained on the UCI Heart Disease "
        "dataset and is intended for educational and "
        "demonstration purposes."
    )


# ============================================================
# HERO HEADER
# ============================================================

st.html("""
<div class="hero">
    <div class="hero-title">
        ❤️ Heart Disease Prediction System
    </div>

    <div class="hero-subtitle">
        Enter available clinical information to generate
        a machine-learning based multiclass prediction.
        Missing values can be left unprovided.
    </div>
</div>
""")


# ============================================================
# PATIENT INPUT FORM
# ============================================================

st.markdown(
    '<div class="section-title">Patient Assessment</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Provide the available patient information below. '
    'Fields that are unavailable may be left as '
    '"Not provided".'
    '</div>',
    unsafe_allow_html=True
)


with st.form("patient_form"):

    # --------------------------------------------------------
    # DEMOGRAPHICS
    # --------------------------------------------------------

    st.markdown("### 👤 Demographics")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=None,
            placeholder="Enter age"
        )

    with col2:

        sex = st.selectbox(
            "Sex",
            options=[
                "Not provided",
                "Male",
                "Female"
            ],
            index=0
        )


    # --------------------------------------------------------
    # CARDIAC SYMPTOMS
    # --------------------------------------------------------

    st.markdown("### 🫀 Cardiac Symptoms")

    col1, col2 = st.columns(2)

    with col1:

        cp = st.selectbox(
            "Chest Pain Type",
            options=[
                "Not provided",
                "typical angina",
                "atypical angina",
                "non-anginal",
                "asymptomatic"
            ],
            index=0
        )

    with col2:

        exang = st.selectbox(
            "Exercise Induced Angina",
            options=[
                "Not provided",
                True,
                False
            ],
            index=0
        )


    # --------------------------------------------------------
    # VITAL SIGNS
    # --------------------------------------------------------

    st.markdown("### 🩺 Vital Signs")

    col1, col2, col3 = st.columns(3)

    with col1:

        trestbps = st.number_input(
            "Resting Blood Pressure",
            min_value=0.0,
            value=None,
            placeholder="mm Hg"
        )

    with col2:

        thalch = st.number_input(
            "Maximum Heart Rate",
            min_value=0.0,
            value=None,
            placeholder="bpm"
        )

    with col3:

        oldpeak = st.number_input(
            "ST Depression (Oldpeak)",
            value=None,
            placeholder="Enter oldpeak"
        )


    # --------------------------------------------------------
    # LABORATORY MEASUREMENTS
    # --------------------------------------------------------

    st.markdown("### 🧪 Laboratory Measurements")

    col1, col2 = st.columns(2)

    with col1:

        chol = st.number_input(
            "Cholesterol",
            min_value=0.0,
            value=None,
            placeholder="mg/dl"
        )

    with col2:

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[
                "Not provided",
                True,
                False
            ],
            index=0
        )


    # --------------------------------------------------------
    # DIAGNOSTIC FINDINGS
    # --------------------------------------------------------

    st.markdown("### 🔬 Diagnostic Findings")

    col1, col2 = st.columns(2)

    with col1:

        restecg = st.selectbox(
            "Resting ECG",
            options=[
                "Not provided",
                "normal",
                "lv hypertrophy",
                "st-t abnormality"
            ],
            index=0
        )

        slope = st.selectbox(
            "ST Segment Slope",
            options=[
                "Not provided",
                "upsloping",
                "flat",
                "downsloping"
            ],
            index=0
        )

    with col2:

        ca = st.selectbox(
            "Number of Major Vessels (CA)",
            options=[
                "Not provided",
                0,
                1,
                2,
                3
            ],
            index=0
        )

        thal = st.selectbox(
            "Thalassemia",
            options=[
                "Not provided",
                "normal",
                "fixed defect",
                "reversable defect"
            ],
            index=0
        )


    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "🔍  Generate Heart Disease Prediction",
        type="primary",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    patient_data = {

        "age": age,

        "sex": sex,

        "cp": cp,

        "trestbps": trestbps,

        "chol": chol,

        "fbs": fbs,

        "restecg": restecg,

        "thalch": thalch,

        "exang": exang,

        "oldpeak": oldpeak,

        "slope": slope,

        "ca": ca,

        "thal": thal
    }


    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    provided_values = sum(
        value not in [None, "Not provided", np.nan]
        for value in patient_data.values()
    )

    if provided_values == 0:

        st.warning(
            "Please provide at least one patient "
            "measurement before generating a prediction."
        )

        st.stop()


    # --------------------------------------------------------
    # RUN MODEL
    # --------------------------------------------------------

    prediction, probabilities = predict_patient(
        patient_data
    )

    prediction_label = class_labels[prediction]

    predicted_probability = (
        probabilities[
            list(target_classes).index(prediction)
        ] * 100
    )


    # ========================================================
    # RESULT SECTION
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">Prediction Result</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Result Card
    # --------------------------------------------------------

    if prediction == 0:

        st.success(
            f"Prediction: {prediction_label}"
        )

    else:

        st.warning(
            f"Prediction: {prediction_label}"
        )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    Class {prediction}
                </div>
                <div class="metric-label">
                    Predicted Class
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {predicted_probability:.2f}%
                </div>
                <div class="metric-label">
                    Model Probability
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {provided_values}/13
                </div>
                <div class="metric-label">
                    Fields Provided
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # PROBABILITY ANALYSIS
    # ========================================================

    st.markdown("### 📊 Class Probability Analysis")

    probability_data = pd.DataFrame({

        "Class": [
            class_labels[class_id]
            for class_id in target_classes
        ],

        "Probability": [
            probability * 100
            for probability in probabilities
        ]
    })

    probability_data["Probability"] = (
        probability_data["Probability"]
        .round(2)
    )


    # --------------------------------------------------------
    # Probability Table
    # --------------------------------------------------------

    display_table = probability_data.copy()

    display_table["Probability"] = (
        display_table["Probability"]
        .apply(lambda x: f"{x:.2f}%")
    )

    st.dataframe(
        display_table,
        hide_index=True,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Probability Chart
    # --------------------------------------------------------

    class_order = [
        class_labels[class_id]
        for class_id in target_classes
    ]

    chart = (
        alt.Chart(probability_data)
        .mark_bar(
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5
        )
        .encode(

            x=alt.X(
                "Class:N",
                sort=class_order,
                title=None,
                axis=alt.Axis(
                    labelAngle=-20
                )
            ),

            y=alt.Y(
                "Probability:Q",
                title="Probability (%)",
                scale=alt.Scale(
                    domain=[0, 100]
                )
            ),

            tooltip=[
                alt.Tooltip(
                    "Class:N",
                    title="Class"
                ),

                alt.Tooltip(
                    "Probability:Q",
                    title="Probability",
                    format=".2f"
                )
            ]
        )
        .properties(
            height=400
        )
    )

    st.altair_chart(
        chart,
        use_container_width=True
    )


    # ========================================================
    # PATIENT INPUT SUMMARY
    # ========================================================

    with st.expander("📋 View Submitted Patient Information"):

        summary_data = pd.DataFrame({
            "Feature": list(patient_data.keys()),
            "Value": list(patient_data.values())
        })

        summary_data["Value"] = (
            summary_data["Value"]
            .replace({
                np.nan: "Not provided",
                None: "Not provided"
            })
        )

        st.dataframe(
            summary_data,
            hide_index=True,
            use_container_width=True
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

with st.expander("⚙️ Model & Technical Information"):

    info_col1, info_col2 = st.columns(2)

    with info_col1:

        st.markdown("#### Model")

        st.write(
            "**Algorithm:** "
            "HistGradientBoostingClassifier"
        )

        st.write(
            "**Task:** Multiclass Classification"
        )

        st.write(
            "**Number of Classes:** "
            f"{len(target_classes)}"
        )

    with info_col2:

        st.markdown("#### Features Used")

        st.write(
            ", ".join(feature_columns)
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown("""
<div class="disclaimer">

<b>⚠️ Important Notice</b><br><br>

This application is an educational machine-learning
demonstration and is <b>not a medical diagnostic tool</b>.
Its predictions are generated from patterns learned from
the training dataset and should not be interpreted as a
clinical diagnosis or a substitute for professional medical
advice.

For medical concerns, consult a qualified healthcare
professional.

</div>
""", unsafe_allow_html=True)