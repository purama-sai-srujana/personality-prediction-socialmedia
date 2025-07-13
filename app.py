from PIL import Image
import pytesseract
import streamlit as st


# (Optional) If you're on Windows and installed Tesseract manually:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.markdown("### 🖼️ Upload an Image with Text (Screenshot of Post)")
uploaded_image = st.file_uploader("Upload an image (PNG or JPG):", type=["png", "jpg", "jpeg"], key="image")

if uploaded_image:
    try:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Extract text from image
        extracted_text = pytesseract.image_to_string(image)

        st.markdown("#### 🔍 Extracted Text from Image:")
        st.text_area("Text found in image:", value=extracted_text, height=150)

        if st.button("🧠 Predict Personality from Image Text"):
            if extracted_text.strip():
                result = predict_personality(extracted_text)
                if result == "happy":
                    st.success("🎉 Personality Prediction: **HAPPY** 😊")
                else:
                    st.error("😢 Personality Prediction: **SAD**")
            else:
                st.warning("No readable text found in the image.")

    except Exception as e:
        st.error(f"Error reading image: {e}")
