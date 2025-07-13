# ---------------------- IMPORTS ----------------------
import streamlit as st
import re
import pickle
import pandas as pd
import docx
import PyPDF2
from io import StringIO
from PIL import Image
import pytesseract
import pytesseract

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 🧠 If you're on Windows and installed Tesseract manually:
# Uncomment and update this path if needed:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ---------------------- LOAD MODEL ----------------------
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# ---------------------- HELPER FUNCTIONS ----------------------

# Clean text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

# Predict
def predict_personality(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    return model.predict(vector)[0]

# Read DOCX
def extract_docx_text(file):
    doc = docx.Document(file)
    return [para.text for para in doc.paragraphs if para.text.strip()]

# Read PDF
def extract_pdf_text(file):
    reader = PyPDF2.PdfReader(file)
    text = []
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text.extend(content.splitlines())
    return [line for line in text if line.strip()]

# ---------------------- UI SETUP ----------------------
st.set_page_config(page_title="🧠 Personality Predictor", page_icon="💬", layout="centered")

st.title("🧠 Personality Prediction from Social Media Posts")
st.subheader("🔍 Predict if the personality is **Happy** 😊 or **Sad** 😢")
st.markdown("Upload a file, paste a post, or upload an image of text. The AI will analyze and predict personality.")

# ---------------------- MANUAL INPUT ----------------------
st.markdown("### ✍️ Type or Paste a Social Media Post")
user_input = st.text_area("Enter a single social media post here:")

if st.button("Predict Text"):
    if user_input.strip():
        result = predict_personality(user_input)
        if result == "happy":
            st.success("🎉 Personality Prediction: **HAPPY** 😊")
        else:
            st.error("😢 Personality Prediction: **SAD**")
    else:
        st.warning("Please enter some text to analyze.")

# ---------------------- FILE UPLOAD ----------------------
st.markdown("---")
st.markdown("### 📁 Upload a Text-based File")

file = st.file_uploader("Upload .txt, .csv (with 'text' column), .docx, or .pdf", type=["txt", "csv", "docx", "pdf"])

if file:
    try:
        ext = file.name.split('.')[-1].lower()

        if ext == "csv":
            df = pd.read_csv(file)
            if 'text' in df.columns:
                df['Prediction'] = df['text'].apply(predict_personality)
                st.success("✅ Predictions completed.")
                st.dataframe(df[['text', 'Prediction']])
            else:
                st.error("CSV must contain a 'text' column.")

        elif ext == "txt":
            lines = file.read().decode("utf-8").splitlines()
            results = [{"Post": line, "Prediction": predict_personality(line)} for line in lines if line.strip()]
            st.success("✅ Predictions completed.")
            st.dataframe(results)

        elif ext == "docx":
            lines = extract_docx_text(file)
            results = [{"Post": line, "Prediction": predict_personality(line)} for line in lines]
            st.success("✅ Predictions completed.")
            st.dataframe(results)

        elif ext == "pdf":
            lines = extract_pdf_text(file)
            results = [{"Post": line, "Prediction": predict_personality(line)} for line in lines]
            st.success("✅ Predictions completed.")
            st.dataframe(results)

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

# ---------------------- IMAGE UPLOAD ----------------------
st.markdown("---")
st.markdown("### 🖼️ Upload an Image (Screenshot of Post or Text)")

image_file = st.file_uploader("Upload PNG or JPG:", type=["png", "jpg", "jpeg"], key="img")

if image_file:
    try:
        img = Image.open(image_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        extracted_text = pytesseract.image_to_string(img)
        st.text_area("📃 Text extracted from image:", value=extracted_text, height=150)

        if st.button("Predict Image Text"):
            if extracted_text.strip():
                result = predict_personality(extracted_text)
                if result == "happy":
                    st.success("🎉 Personality Prediction: **HAPPY** 😊")
                else:
                    st.error("😢 Personality Prediction: **SAD**")
            else:
                st.warning("No readable text found in image.")
    except Exception as e:
        st.error(f"⚠️ Error reading image: {e}")

# ---------------------- FOOTER ----------------------
st.markdown("---")
st.caption("📘 Project: Personality Prediction from Social Media Posts")
st.caption("👩‍💻 Developer: PURAMA SAI SRUJANA | Dept. of CSE | 2025")
