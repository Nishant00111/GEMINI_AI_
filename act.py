import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="Chat with PDF", page_icon="📚")

# add sql databse functions.

import mysql.connector

def connect_db():
    """Establishes a connection to the MySQL database."""
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",  # your username
            password="Nishant@123",  # your password
            database="chat"  # your database name
        )
        cursor = conn.cursor()
        return conn, cursor
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None, None

def init_db():
    """Creates the chat_history table in MySQL if it does not exist."""
    conn, cursor = connect_db()  # Ensure this is unpacked correctly
    if conn is not None and cursor is not None:
        cursor.execute(''' 
            CREATE TABLE IF NOT EXISTS chat_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    else:
        print("Failed to connect to the database.")






def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    answer = response["output_text"]
    
    # Save the question and answer to MySQL
    conn, cursor = connect_db()
    cursor.execute("INSERT INTO chat_history (question, answer) VALUES (%s, %s)", (user_question, answer))
    conn.commit()
    conn.close()

    st.write("Reply: ", answer)


def fetch_chat_history():
    """Fetch the last 10 chat history entries from MySQL."""
    conn, cursor = connect_db()
    cursor.execute("SELECT question, answer FROM chat_history ORDER BY timestamp DESC LIMIT 10")
    history = cursor.fetchall()
    conn.close()
    return history


with st.sidebar:
    st.title("Chat History")
    history = fetch_chat_history()
    for q, a in history:
        with st.expander(q):
            st.write(a)



def main():
    st.header("Chat with PDF using Gemini💁")

    # Ensure MySQL table exists
    init_db()  

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Display chat history in sidebar
    with st.sidebar:
        st.title("Chat History")
        history = fetch_chat_history()
        for q, a in history:
            with st.expander(q):
                st.write(a)

if __name__ == "__main__":
    main()