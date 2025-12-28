import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
import pdfplumber
from io import BytesIO

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AutoInsight AI", layout="wide", page_icon="🚀")

# ---------------- LOGO AND HEADER ----------------
with open("logo.txt", "r") as f:
    logo = f.read()
st.code(logo, language="text")

st.title("AutoInsight AI — LLM Powered Data Analyst")
st.caption("Upload CSV, Excel, or PDF → Get instant insights, charts & AI-generated analysis")

st.markdown("""
<style>
h1 {text-align:left; color:#4A90E2;}
.stButton>button {background-color:#4A90E2; color:white; font-size:16px; border-radius:10px;}
.sidebar .sidebar-content {background-color:#f0f2f6;}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write("AutoInsight AI uses AI to analyze your data and provide insights.")
    st.write("**Supported Formats:** CSV, Excel (.xlsx), PDF (with tables)")
    st.write("**AI Model:** FLAN-T5 Small (Hugging Face API)")
    st.markdown("---")
    st.write("Built with Streamlit & Transformers")
# ---------------- LOAD AI MODEL ----------------
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except:
    HF_TOKEN = os.getenv("HF_TOKEN") or "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Fallback for local

HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# ---------------- FUNCTIONS ----------------
def generate_ai_insights(summary_text):
    # Truncate if too long
    if len(summary_text) > 1500:
        summary_text = summary_text[:1500] + "..."
    
    prompt = f"""
You are an expert data analyst. Analyze this dataset summary and provide actionable insights.

DATASET ANALYSIS:
{summary_text}

Provide:
1. Key trends and patterns
2. Anomalies or outliers  
3. Business insights
4. Recommendations

Keep response under 300 words.
"""
    
    try:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.3,
                "do_sample": True
            }
        }
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and result:
                return result[0].get("generated_text", "No insights generated.")
            return str(result)
        else:
            return f"API Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

# ---------------- UI ----------------
uploaded_file = st.file_uploader("📂 Upload CSV, Excel (.xlsx), or PDF file", type=["csv", "xlsx", "pdf"])

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_type == 'xlsx':
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif file_type == 'pdf':
            with pdfplumber.open(BytesIO(uploaded_file.getvalue())) as pdf:
                tables = []
                for page in pdf.pages:
                    table = page.extract_table()
                    if table:
                        tables.append(table)
                if tables:
                    # Combine all tables, assume first row of first table is header
                    combined_table = []
                    for table in tables:
                        combined_table.extend(table)
                    df = pd.DataFrame(combined_table[1:], columns=combined_table[0])
                else:
                    st.error("No tables found in the PDF. Please ensure the PDF contains tabular data.")
                    st.stop()
        else:
            st.error("Unsupported file type. Please upload CSV, Excel, or PDF.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.success(f"File '{uploaded_file.name}' loaded successfully! Shape: {df.shape}")

    # ---------------- DATA EXPLORATION ----------------
    tab1, tab2, tab3, tab4 = st.tabs(["Preview", "Statistics", "Visualizations", "AI Insights"])

    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))
        st.write(f"**Total Rows:** {df.shape[0]} | **Total Columns:** {df.shape[1]}")

    with tab2:
        st.subheader("Basic Statistics")
        st.dataframe(df.describe())
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            st.subheader("Missing Values")
            st.dataframe(missing[missing > 0])
        else:
            st.info("No missing values detected!")

    with tab3:
        st.subheader("Visualizations")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                selected_col = st.selectbox("Select numeric column for histogram", numeric_cols)
                fig, ax = plt.subplots()
                sns.histplot(df[selected_col], kde=True, ax=ax)
                st.pyplot(fig)
            
            with col2:
                if len(numeric_cols) > 1:
                    x_col = st.selectbox("X-axis", numeric_cols, key="x")
                    y_col = st.selectbox("Y-axis", numeric_cols, key="y")
                    fig, ax = plt.subplots()
                    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
                    st.pyplot(fig)
            
            # Correlation heatmap
            if len(numeric_cols) > 1:
                st.subheader("Correlation Heatmap")
                corr = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
        
        if categorical_cols:
            st.subheader("Categorical Data")
            cat_col = st.selectbox("Select categorical column for bar chart", categorical_cols)
            fig, ax = plt.subplots()
            df[cat_col].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)

    with tab4:
        # Enhanced data analysis for AI insights
        analysis_parts = []
        analysis_parts.append(f"Dataset Summary:\n- Columns: {list(df.columns)}\n- Rows: {df.shape[0]}\n- Missing values: {df.isnull().sum().sum()}\n- Data types: {df.dtypes.to_dict()}")
        
        # Numeric analysis
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            analysis_parts.append(f"\nNumeric Columns Statistics:\n{df[numeric_cols].describe().to_string()}")
            
            # Correlations
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                analysis_parts.append(f"\nCorrelations:\n{corr.to_string()}")
            
            # Outliers detection using IQR
            outliers_info = []
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                if len(outliers) > 0:
                    outliers_info.append(f"{col}: {len(outliers)} outliers")
            if outliers_info:
                analysis_parts.append(f"\nPotential Outliers (IQR method):\n" + "\n".join(outliers_info))
            
            # Skewness
            skewness = df[numeric_cols].skew()
            analysis_parts.append(f"\nSkewness:\n{skewness.to_string()}")
        
        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            cat_analysis = []
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                cat_analysis.append(f"{col}:\n{value_counts.head(5).to_string()}")
            analysis_parts.append(f"\nCategorical Columns (Top 5 values):\n" + "\n\n".join(cat_analysis))
        
        summary_text = "\n".join(analysis_parts)

        if st.button("Generate AI Insights"):
            with st.spinner("Analyzing with AI..."):
                insights = generate_ai_insights(summary_text)
            st.subheader("AI Generated Insights")
            st.success(insights)

else:
    st.info("Upload a CSV, Excel, or PDF file to begin analysis.")
