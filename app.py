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

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(selected_pdfs):
    """Extract text from selected PDFs."""
    text = ""
    for pdf in selected_pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Create a vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Set up the conversational chain with a prompt template."""
    prompt_template = """
    You are an AI study assistant. Keep answers *short, precise, and clear*.
    - *Use the document context* to answer.
    - *If the answer is not in the document, say:* "This information is not available."
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
    """Handle user questions and provide responses based on the vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("📝 *Reply:*", response["output_text"])

def main():
    st.set_page_config(page_title="Chat with PDF")
    st.header("📚 Chat with Your PDFs using Gemini")

    # Initialize session state for uploaded files and their selection
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'selected_files' not in st.session_state:
        st.session_state.selected_files = []

    # File uploader in the sidebar
    with st.sidebar:
        st.title("📂 Upload & Select PDFs")
        uploaded_files = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=['pdf'])

        if uploaded_files:
            # Update session state with newly uploaded files
            st.session_state.uploaded_files.extend(uploaded_files)

            # Display checkboxes for each uploaded file
            st.subheader("✅ Select PDFs to Process")
            for file in st.session_state.uploaded_files:
                if file.name not in st.session_state.selected_files:
                    st.session_state.selected_files.append(file.name)
                selected = st.checkbox(file.name, value=True, key=file.name)
                if not selected and file.name in st.session_state.selected_files:
                    st.session_state.selected_files.remove(file.name)

        # Button to process selected PDFs
        if st.button("Submit & Process"):
            if not st.session_state.selected_files:
                st.warning("⚠ Please select at least one PDF!")
            else:
                selected_pdfs = [file for file in st.session_state.uploaded_files if file.name in st.session_state.selected_files]
                with st.spinner("🔄 Processing selected PDFs..."):
                    raw_text = get_pdf_text(selected_pdfs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("✅ Processing complete! Now ask your question.")
                    else:
                        st.error("❌ No text could be extracted from the selected PDFs.")

    # Input for user questions
    user_question = st.text_input("Ask a question from the selected PDF files:")
    if user_question:
        user_input(user_question)

if _name_ == "_main_":
    main()
