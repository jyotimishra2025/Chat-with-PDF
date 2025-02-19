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
    You are an AI-powered study assistant designed to help students understand and learn from their PDFs. 
    Your responses should be *clear, detailed, and structured*, making it easy for students to grasp concepts.  
    Follow these rules when answering questions:

    1️⃣ *Use the provided context* to generate answers. If the exact answer isn't available, summarize the closest relevant information.
    2️⃣ *Explain concepts step-by-step*, especially for technical or theoretical questions.
    3️⃣ *Provide definitions and examples* where needed to simplify complex topics.
    4️⃣ *Break down long answers into bullet points or numbered lists* for better readability.
    5️⃣ *If the question requires reasoning*, show your thought process logically.
    6️⃣ *If a direct answer is not in the document*, respond with:  
       "The exact answer is not in the document, but here’s what I found that might help:"  
       Then, try to provide a related explanation.
    7️⃣ *For large text answers, include a short summary at the end*.

    ---
    📖 *Context from the document:*  
    {context}

    ❓ *Student's Question:*  
    {question}

    📝 *AI's Response (Well-structured, simple, and helpful):*
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2, max_tokens=1500)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Increase the number of retrieved documents from FAISS
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=5)  # Retrieve top 5 similar chunks instead of default 3

    chain = get_conversational_chain()
    
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Chat PDF")
    st.header("AI PDF Assistant 🤖📄")

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




if __name__ == "__main__":
    main()
