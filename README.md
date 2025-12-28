# AutoInsight AI

🚀 **AI-Powered Data Analysis Tool**

Upload your datasets (CSV, Excel, PDF) and get instant insights, visualizations, and AI-generated analysis using local LLM.

## Features

- 📊 Multi-format file support (CSV, Excel, PDF)
- 📈 Interactive visualizations and statistics
- 🤖 AI-powered insights with FLAN-T5 model
- 🎨 Modern tabbed interface
- 📋 Data quality analysis

## Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deployment Options

### Option 1: Hugging Face Spaces (Recommended for ML apps)
1. Create a Hugging Face account
2. Go to Spaces → Create new Space
3. Choose Streamlit as framework
4. Upload your files (app.py, requirements.txt, logo.txt)
5. Spaces automatically handles transformers models

### Option 2: Streamlit Cloud
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repo
4. Deploy (Note: May have issues with large model loading)

### Option 3: Heroku
1. Create Heroku app
2. Add buildpacks for Python
3. Deploy via Git or CLI

## Requirements

- Python 3.8+
- 4GB+ RAM recommended for model loading
- Internet connection for initial model download

## Tech Stack

- **Frontend**: Streamlit
- **AI**: Transformers (FLAN-T5-Small)
- **Data**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **PDF Processing**: pdfplumber

## License

MIT License
