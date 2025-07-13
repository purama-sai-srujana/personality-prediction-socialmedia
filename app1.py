import streamlit as st
import re
import pickle
import pandas as pd
import docx
import PyPDF2
from io import StringIO

# Load the trained model and vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Clean the input text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

# Predict personality
def predict_personality(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    return model.predict(vector)[0]

# Extract text from Word document
def extract_docx_text(file):
    doc = docx.Document(file)
    return [para.text for para in doc.paragraphs if para.text.strip()]

# Extract text from PDF
def extract_pdf_text(file):
    reader = PyPDF2.PdfReader(file)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.extend(page_text.splitlines())
    return [line for line in text if line.strip()]

# Streamlit Interface
st.set_page_config(page_title="🧠 Personality Prediction", page_icon="🔮")
st.title("🔮 Personality Prediction from Social Media Posts")
st.subheader("📱 Analyze what your words reveal about your personality")

st.markdown("Enter a social media post or upload a file (.csv, .txt, .docx, .pdf) to predict personality as **Happy** or **Sad**.")

# Manual input
st.markdown("### ✍️ Predict from a Single Post")
user_input = st.text_area("Enter a social media post:")

if st.button("🔍 Predict Personality"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a post.")
    else:
        result = predict_personality(user_input)
        if result == "happy":
            st.success("🎉 Personality Prediction: **HAPPY** 😊")
        else:
            st.error("😢 Personality Prediction: **SAD**")

# File upload section
st.markdown("---")
st.markdown("### 📂 Upload a File (.txt, .csv, .docx, .pdf)")

uploaded_file = st.file_uploader("Upload a file:", type=["txt", "csv", "docx", "pdf"])

if uploaded_file:
    try:
        file_type = uploaded_file.name.split('.')[-1].lower()

        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
            if 'text' in df.columns:
                df['Prediction'] = df['text'].apply(predict_personality)
                st.success("✅ Predictions completed.")
                st.dataframe(df[['text', 'Prediction']])
            else:
                st.error("❌ CSV must have a column named 'text'.")

        elif file_type == "txt":
            text_lines = uploaded_file.read().decode("utf-8").splitlines()
            results = [{"Post": line, "Prediction": predict_personality(line)} for line in text_lines if line.strip()]
            st.success("✅ Predictions completed.")
            st.dataframe(results)

        elif file_type == "docx":
            text_lines = extract_docx_text(uploaded_file)
            results = [{"Post": line, "Prediction": predict_personality(line)} for line in text_lines]
            st.success("✅ Predictions completed.")
            st.dataframe(results)

        elif file_type == "pdf":
            text_lines = extract_pdf_text(uploaded_file)
            results = [{"Post": line, "Prediction": predict_personality(line)} for line in text_lines]
            st.success("✅ Predictions completed.")
            st.dataframe(results)

        else:
            st.error("❌ Unsupported file format.")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

# Footer
st.markdown("---")
st.caption("📘 Project: Personality Prediction from Social Media Posts")
st.caption("👩‍💻 Developed by: PURAMA SAI SRUJANA (CSE, 2025)")
