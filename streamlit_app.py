import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoTokenizer
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
    st.write("**AI Model:** FLAN-T5 Small (local inference)")
    st.markdown("---")
    st.write("Built with Streamlit & Transformers")
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline(
        "text2text-generation",
        model=model_name,
        tokenizer=tokenizer,
        max_length=512
    )
    return pipe, tokenizer

llm, tokenizer = load_llm()

# ---------------- FUNCTIONS ----------------
def generate_ai_insights(summary_text, tokenizer):
    # Truncate input if too long
    tokens = tokenizer.encode(summary_text, add_special_tokens=True)
    if len(tokens) > 450:  # Leave room for prompt
        truncated_tokens = tokens[:450]
        summary_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    prompt = f"""
You are an expert data analyst with 10+ years of experience. Analyze the following dataset summary and provide deep, actionable insights.

DATASET ANALYSIS:
{summary_text}

INSTRUCTIONS:
1. **Key Trends & Patterns**: Identify significant trends, correlations, and patterns in the data. Explain what they mean.
2. **Anomalies & Outliers**: Highlight any anomalies, outliers, or unusual data points and their potential causes.
3. **Business Insights**: Provide 4-5 specific, actionable business insights based on the data analysis.
4. **Recommendations**: Suggest 3-4 concrete actions or next steps for data-driven decision making.
5. **Data Quality Notes**: Comment on data quality, missing values, and any concerns.

FORMAT:
- Use clear headings for each section
- Be specific and reference actual data points/numbers
- Focus on actionable insights, not generic statements
- Keep response concise but comprehensive (300-500 words)

Respond as a professional data analyst would.
"""
    output = llm(prompt)[0]["generated_text"]
    return output

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
                insights = generate_ai_insights(summary_text, tokenizer)
            st.subheader("AI Generated Insights")
            st.success(insights)

else:
    st.info("Upload a CSV, Excel, or PDF file to begin analysis.")
