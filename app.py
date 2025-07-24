import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import torch
 
 
# Title
st.set_page_config(page_title="📄 PDF Summarizer")
st.title("📄 PDF Summarizer")
 
# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
 
# Extract text from PDF
def extract_text_from_pdf(file):
    pdf_text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            pdf_text += page.get_text()
    return pdf_text
 
# Load summarization pipeline
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
 
# Summarize button
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    st.subheader("📃 Extracted Text")
    st.text_area("Text from PDF", text, height=200)
 
    if st.button("🔍 Summarize"):
        if len(text.strip()) == 0:
            st.warning("No text found in the PDF.")
        else:
            summarizer = load_summarizer()
            # You may need to chunk long texts (many models have a 1024-token limit)
            summary = summarizer(text[:1024], max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
            st.subheader("📝 Summary")
            st.write(summary)