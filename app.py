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
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(selected_pdfs):
    """Extract text only from selected PDFs."""
    text = ""
    for pdf in selected_pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are an AI study assistant. Keep answers *short, precise, and clear*.
    - *Use the document context* to answer.
    - *If answer is not in the document, say:* "This information is not available."
    - *Avoid unnecessary details; keep responses **to-the-point*.

    📖 *Context:*  
    {context}

    ❓ *User's Question:*  
    {question}

    📝 *AI Response:*
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, max_tokens=800)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("📝 *Reply:*", response["output_text"])

def main():
    st.set_page_config(page_title="Chat with PDF")
    st.header("📚 Chat with Your PDFs using Gemini")

    user_question = st.text_input("Ask a question from the selected PDF files:")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 Upload & Select PDFs")
        pdf_docs = st.file_uploader("Upload up to 5 PDF Files", accept_multiple_files=True)

        selected_pdfs = []
        if pdf_docs:
            st.subheader("✅ Select PDFs to Process")
            check_states = {}
            for pdf in pdf_docs:
                check_states[pdf.name] = st.checkbox(pdf.name, value=True)
            
            selected_pdfs = [pdf for pdf in pdf_docs if check_states[pdf.name]]

        if st.button("Submit & Process"):
            if len(selected_pdfs) == 0:
                st.warning("⚠ Please select at least one PDF!")
            else:
                with st.spinner("🔄 Processing selected PDFs..."):
                    raw_text = get_pdf_text(selected_pdfs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Processing complete! Now ask your question.")

if _name_ == "_main_":
    main()
