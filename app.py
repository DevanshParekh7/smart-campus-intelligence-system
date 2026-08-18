import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Smart Campus Intelligence System",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Smart Campus Intelligence System")
st.markdown("### AI-Based Student Academic Performance Analysis & Dashboard")

# 1. Dataset Loading
@st.cache_data
def load_default_data():
    try:
        return pd.read_csv("Students_Performance_dataset.csv")
    except FileNotFoundError:
        return None

uploaded_file = st.sidebar.file_uploader("Upload Student Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_default_data()

if df is None:
    st.info("Please upload a `Students_Performance_dataset.csv` file from the sidebar to proceed.")
    st.stop()

# 2. Data Cleaning
df_clean = df.copy()
df_clean.columns = df_clean.columns.str.strip()

for col in df_clean.select_dtypes(include="object").columns:
    df_clean[col] = df_clean[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# 3. High-Level Metrics
st.subheader("Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Students", f"{df_clean.shape[0]:,}")
col2.metric("Total Features", f"{df_clean.shape[1]}")
col3.metric("Average CGPA", f"{df_clean['What is your current CGPA?'].mean():.2f}")
col4.metric("Average Previous SGPA", f"{df_clean['What was your previous SGPA?'].mean():.2f}")

st.divider()

# 4. Exploratory Data Views
tab1, tab2, tab3 = st.tabs(["📊 Distributions & Correlations", "📋 Dataset Overview", "📈 Categorical Analysis"])

with tab1:
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("**Distribution of Previous SGPA**")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(df_clean["What was your previous SGPA?"], kde=True, ax=ax, color="#1f77b4")
        ax.set_xlabel("Previous SGPA")
        ax.set_ylabel("Number of Students")
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        st.markdown("**Distribution of Current CGPA**")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(df_clean["What is your current CGPA?"], kde=True, ax=ax, color="#2ca02c")
        ax.set_xlabel("Current CGPA")
        ax.set_ylabel("Number of Students")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**SGPA vs CGPA Correlation**")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.scatterplot(
        data=df_clean,
        x="What was your previous SGPA?",
        y="What is your current CGPA?",
        hue="Gender" if "Gender" in df_clean.columns else None,
        ax=ax
    )
    ax.set_title("Previous SGPA vs Current CGPA")
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    st.markdown("**Raw / Cleaned Data Preview**")
    st.dataframe(df_clean, use_container_width=True)
    
    st.markdown("**Statistical Summary**")
    st.dataframe(df_clean.describe(include="all").T, use_container_width=True)

with tab3:
    cat_cols = df_clean.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        selected_cat = st.selectbox("Select Categorical Feature to Inspect", cat_cols)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        top_cats = df_clean[selected_cat].value_counts().head(10)
        sns.barplot(x=top_cats.index, y=top_cats.values, ax=ax, palette="viridis")
        plt.xticks(rotation=45, ha="right")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)