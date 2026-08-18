import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Smart Campus Intelligence System",
    page_icon="🎓",
    layout="wide"
)

MODEL_FILE = "smart_campus_model.joblib"
DATA_FILE = "smart_campus_data.csv"
FEATURE_FILE = "smart_campus_features.joblib"

st.title("🎓 Smart Campus Intelligence System")
st.caption("AI-based Student Academic Performance Analysis & Prediction")

# -----------------------------
# Load project artifacts
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

@st.cache_resource
def load_features():
    return joblib.load(FEATURE_FILE)

if not all(Path(x).exists() for x in [MODEL_FILE, DATA_FILE, FEATURE_FILE]):
    st.error(
        "Project files are missing. Place smart_campus_model.joblib, "
        "smart_campus_data.csv and smart_campus_features.joblib in the "
        "same folder as app.py."
    )
    st.stop()

model = load_model()
data = load_data()
feature_columns = load_features()

# -----------------------------
# Risk function
# -----------------------------
def academic_risk(cgpa):
    if cgpa < 5.0:
        return "High Risk"
    elif cgpa < 6.5:
        return "Moderate Risk"
    return "Low Risk"

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Module",
    [
        "Campus Overview",
        "Student Prediction",
        "Prediction from CSV",
        "Model Information"
    ]
)

# -----------------------------
# Campus Overview
# -----------------------------
if page == "Campus Overview":
    st.header("📊 Campus Academic Overview")

    cgpa_col = "What is your current CGPA?"
    sgpa_col = "What was your previous SGPA?"
    attendance_col = "Average attendance on class"

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Students", f"{len(data):,}")
    c2.metric("Average CGPA", f"{data[cgpa_col].mean():.2f}")
    c3.metric("Average Previous SGPA", f"{data[sgpa_col].mean():.2f}")
    c4.metric("Median CGPA", f"{data[cgpa_col].median():.2f}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Current CGPA Distribution")
        st.bar_chart(
            data[cgpa_col].value_counts(bins=10).sort_index()
        )

    with right:
        st.subheader("Previous SGPA vs Current CGPA")
        chart_df = data[[sgpa_col, cgpa_col]].copy()
        chart_df.columns = ["Previous SGPA", "Current CGPA"]
        st.scatter_chart(chart_df, x="Previous SGPA", y="Current CGPA")

    st.subheader("Academic Summary")
    summary = pd.DataFrame({
        "Metric": [
            "Highest CGPA",
            "Lowest CGPA",
            "Average CGPA",
            "Students below 5.0",
            "Students between 5.0 and 6.49",
            "Students at/above 6.5"
        ],
        "Value": [
            round(data[cgpa_col].max(), 2),
            round(data[cgpa_col].min(), 2),
            round(data[cgpa_col].mean(), 2),
            int((data[cgpa_col] < 5.0).sum()),
            int(((data[cgpa_col] >= 5.0) & (data[cgpa_col] < 6.5)).sum()),
            int((data[cgpa_col] >= 6.5).sum())
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

# -----------------------------
# Single student prediction
# -----------------------------
elif page == "Student Prediction":
    st.header("🔮 Student CGPA Prediction")
    st.write(
        "Enter the student's information. The trained machine-learning "
        "pipeline will preprocess the data and estimate the student's CGPA."
    )

    # Build a dynamic form from the training feature schema.
    values = {}

    numeric_columns = data[feature_columns].select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = data[feature_columns].select_dtypes(
        include=["object"]
    ).columns

    st.subheader("Student Information")

    cols = st.columns(2)

    with st.form("student_form"):
        for i, col in enumerate(feature_columns):
            with cols[i % 2]:
                if col in numeric_columns:
                    default = float(data[col].median())
                    if pd.api.types.is_integer_dtype(data[col]):
                        values[col] = st.number_input(
                            col,
                            value=int(round(default)),
                            step=1
                        )
                    else:
                        values[col] = st.number_input(
                            col,
                            value=default
                        )
                else:
                    options = sorted(
                        data[col].dropna().astype(str).unique().tolist()
                    )
                    values[col] = st.selectbox(col, options)

        submitted = st.form_submit_button("Predict CGPA", type="primary")

    if submitted:
        student_df = pd.DataFrame([values], columns=feature_columns)
        prediction = float(model.predict(student_df)[0])
        risk = academic_risk(prediction)

        st.divider()

        a, b = st.columns(2)
        a.metric("Predicted CGPA", f"{prediction:.2f}")
        b.metric("Academic Risk", risk)

        if risk == "High Risk":
            st.error("The prediction falls in the project-defined High Risk category.")
        elif risk == "Moderate Risk":
            st.warning("The prediction falls in the project-defined Moderate Risk category.")
        else:
            st.success("The prediction falls in the project-defined Low Risk category.")

# -----------------------------
# CSV prediction
# -----------------------------
elif page == "Prediction from CSV":
    st.header("📁 Batch Student Prediction")

    st.write(
        "Upload a CSV containing the same input features used during training. "
        "The system will generate predicted CGPA and project-defined academic risk."
    )

    uploaded = st.file_uploader(
        "Upload student CSV",
        type=["csv"]
    )

    if uploaded is not None:
        input_df = pd.read_csv(uploaded)

        missing = [c for c in feature_columns if c not in input_df.columns]

        if missing:
            st.error("The uploaded CSV is missing required columns:")
            st.write(missing)
        else:
            input_df = input_df[feature_columns].copy()
            predictions = model.predict(input_df)

            result_df = pd.read_csv(uploaded)
            result_df["Predicted_CGPA"] = predictions
            result_df["Academic_Risk"] = [
                academic_risk(x) for x in predictions
            ]

            st.success("Predictions generated successfully.")
            st.dataframe(result_df, use_container_width=True)

            csv = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download Prediction Report",
                data=csv,
                file_name="student_prediction_report.csv",
                mime="text/csv"
            )

# -----------------------------
# Model information
# -----------------------------
elif page == "Model Information":
    st.header("🤖 Machine Learning Model")

    st.write(
        "The dashboard uses the leakage-free trained model produced by the "
        "Smart Campus Intelligence System notebook."
    )

    st.subheader("Prediction Target")
    st.code("What is your current CGPA?")

    st.subheader("Model Pipeline")
    st.markdown("""
    **Input Student Data → Preprocessing → Trained ML Model → Predicted CGPA → Academic Risk**

    The notebook compares multiple regression algorithms and performs
    cross-validation and Random Forest hyperparameter tuning before selecting
    the best-performing model.
    """)

    st.subheader("Academic Risk Rule")
    st.markdown("""
    - **High Risk:** predicted CGPA < 5.0
    - **Moderate Risk:** predicted CGPA 5.0–6.49
    - **Low Risk:** predicted CGPA ≥ 6.5

    These are project-defined decision-support thresholds, not official
    institutional standards.
    """)

    st.subheader("Input Features")
    st.write(feature_columns)
