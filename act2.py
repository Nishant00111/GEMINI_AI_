import streamlit as st
import pandas as pd
import tempfile
from langchain_community.document_loaders import CSVLoader
from google.generativeai import GenerativeModel
import os

# Set up Gemini API Key
gemini_api_key = "AIzaSyB96HtxiYRYRbMEKbVR4YZ7cKcDZAhUm_M"  # Replace with your actual API key
os.environ["GOOGLE_API_KEY"] = gemini_api_key

# Streamlit UI
st.title("CSV Chatbot with LangChain & Gemini")

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(uploaded_file.read())  # Write file content
        temp_file_path = temp_file.name  # Get temp file path

    # Load CSV into Pandas DataFrame
    df = pd.read_csv(temp_file_path)
    st.write("### Data Preview:")
    st.dataframe(df.head())

    # Load CSV into LangChain Document Loader
    loader = CSVLoader(file_path=temp_file_path)
    documents = loader.load()

    # Gemini AI Model
    model = GenerativeModel(model_name="gemini-pro")

    # User Query
    query = st.text_input("Ask a question about the data:")

    if query:
        # Convert part of the dataframe to a string for Gemini context
        data_context = df.head(10).to_string(index=False)  # Show first 10 rows

        # Pass formatted data to Gemini API
        full_prompt = f"Here is a sample of the dataset:\n{data_context}\n\nNow, answer the question: {query}"

        response = model.generate_content(full_prompt)
        st.write("### Response:")
        st.write(response.text)
