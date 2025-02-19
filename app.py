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
    You are an AI-powered study assistant designed to help students learn from PDFs in a friendly and interactive way.  
    Your goal is to provide *clear, engaging, and structured answers* while keeping the conversation fun and informative.  
    Follow these rules when responding:

    🎯 *Make it Conversational:* Start with a friendly acknowledgment like "Great question!" or "Interesting topic!"  
    🎯 *Explain in a Simple & Engaging Way:* Break down complex topics into *easy-to-understand steps.*  
    🎯 *Encourage Curiosity:* Ask a follow-up question at the end like:  
        - "Does this make sense?"  
        - "Would you like a real-world example?"  
        - "Want me to summarize this in one sentence?"  
    🎯 *Use Examples & Analogies:* Relate technical topics to *real-life scenarios* to make learning fun.  
    🎯 *Summarize the Answer:* If the response is long, provide a *quick summary at the end.*  
    🎯 *If the answer is not available in the document:* Say  
        "I couldn’t find the exact answer, but here’s what I found that might help!"  

    ---
    📖 *Context from the document:*  
    {context}

    ❓ *Student's Question:*  
    {question}

    📝 *AI's Response (Interactive, engaging, and structured):*
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, max_tokens=1500)

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
