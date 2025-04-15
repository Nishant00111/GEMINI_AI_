# import streamlit as st
# import pandas as pd
# import tempfile
# import os
# from langchain_experimental.agents import create_csv_agent
# from langchain_google_genai import GoogleGenerativeAI
# from dotenv import load_dotenv
# import google.generativeai as genai

# # Load environment variables
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Streamlit UI
# st.title("CSV Chatbot with LangChain & Gemini")

# # File Upload
# uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# if uploaded_file is not None:
#     # Save uploaded file to a temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
#         temp_file.write(uploaded_file.read())  # Write uploaded file content
#         temp_file_path = temp_file.name  # Store file path

#     # Load CSV into Pandas DataFrame
#     try:
#         df = pd.read_csv(temp_file_path, encoding="utf-8")  # Handle UTF-8 encoding
#     except UnicodeDecodeError:
#         df = pd.read_csv(temp_file_path, encoding="ISO-8859-1")  # Handle encoding issues

#     # Show Data Preview
#     st.write("### Data Preview:")
#     st.dataframe(df.head())

#     # Initialize Gemini Model with correct file path
#     agent = create_csv_agent(
#         GoogleGenerativeAI(temperature=0, model="gemini-pro"),
#         temp_file_path,  # ✅ Correctly passing the uploaded file path
#         verbose=True
#     )

#     # User Query
#     query = st.text_input("Ask a question about the data:")

#     if query:
#         response = agent.run(query)
#         st.write("### Response:")
#         st.write(response)

#     # Cleanup: Delete temporary file after use
#     os.remove(temp_file_path)



# UPDATE THE CODE FOR DRAWING THE QUERIES ALSO ----


import streamlit as st
import pandas as pd
import tempfile
import os
import chardet
import google.generativeai as genai
from langchain_experimental.agents import create_csv_agent
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit UI
st.title("CSV & Excel Chatbot with LangChain & Gemini")

# File Upload
uploaded_file = st.file_uploader("Upload your file", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    file_ext = uploaded_file.name.split(".")[-1]  # Extract file extension

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
        temp_file.write(uploaded_file.read())  # Write file content
        temp_file_path = temp_file.name  # Get temp file path

    # Function to detect encoding
    def detect_encoding(file_path):
        with open(file_path, "rb") as f:
            result = chardet.detect(f.read(100000))  # Read sample for detection
        return result["encoding"] if result["encoding"] else "ISO-8859-1"  # Default encoding

    try:
        if file_ext == "csv":
            encoding = detect_encoding(temp_file_path)  # Auto-detect encoding
            if encoding.lower() not in ["utf-8", "ascii"]:
                encoding = "ISO-8859-1"  # Override encoding for safety
            df = pd.read_csv(temp_file_path, encoding=encoding)
        elif file_ext in ["xls", "xlsx"]:
            df = pd.read_excel(temp_file_path, engine="openpyxl")  # Handle Excel files
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            st.stop()

    except UnicodeDecodeError:
        df = pd.read_csv(temp_file_path, encoding="ISO-8859-1")  # Final fallback

    st.write("### Data Preview:")
    st.dataframe(df.head())

    # ✅ Fix: Read file as binary and re-save it in UTF-8 encoding before passing to agent
    fixed_temp_file = temp_file_path + "_utf8.csv"
    df.to_csv(fixed_temp_file, index=False, encoding="utf-8")  # Convert to UTF-8

    try:
        agent = create_csv_agent(
            GoogleGenerativeAI(temperature=0, model="gemini-pro"),
            fixed_temp_file,  # Use re-encoded file
            verbose=True,
            handle_parsing_errors=True,
            allow_dangerous_code=True  # Allow code execution for plotting
        )

        # User Query
        query = st.text_input("Ask a question about the data:")

        if query:
            response = agent.run(query)
            st.write("### Response:")
            st.write(response)
    
    except Exception as e:
        st.error(f"Error creating agent: {e}")



