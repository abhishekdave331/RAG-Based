import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS              # faiss
from langchain.text_splitter import CharacterTextSplitter       # chunking
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader

genai.configure(api_key="AIzaSyCVrrwxoQ5O6TMMCQL_kbfMW9Cpw9LoXcw")
model = genai.GenerativeModel("gemini-2.0-flash")

# Configure Embedding Model "sentence-transformers/all-MiniLM-L6-v2"
@st.cache_resource(show_spinner="Loading the model..")
def embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Reading the PDF after frontend
st.header("RAG using HF Embeddings + FAISS db")
uploaded_file = st.file_uploader("Upload the Document", type=["pdf"])

if uploaded_file:
    raw_text = ""
    pdf = PdfReader(uploaded_file)
    for index,page in enumerate(pdf.pages):
        context = page.extract_text()
        if context:
            raw_text+=context

    # Chunking using schema
    if raw_text.strip():
        document = Document(page_content=raw_text)
        splitter = CharacterTextSplitter(chunk_size= 1000, chunk_overlap=200)
        chunks = splitter.split_documents([document])

        # HF Embedding
        texts = [chunk.page_content for chunk in chunks]
        vector_db = FAISS.from_texts(texts,embedding_model())
        retriever= vector_db.as_retriever()
        st.markdown("Document processed successfully. Ask questions below..")
        user_input = st.text_input("Enter your query..")

        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            with st.spinner("Analysing the document.."):
                retrieved_doc = retriever.get_relevant_documents(user_input)
                context = "\n\n".join(doc.page_content for doc in retrieved_doc)

                prompt = f"""You are an expert assistant and use the context below to answer the query. 
                            If unsure, just say, 'I don't know'.
                            Context; {context},
                            User query: {user_input}
                            Answer: """
                
                response = model.generate_content(prompt)
                st.markdown("Answer: ")
                st.write(response.text)
else:
    st.warning("Please upload the PDF for review and analysis")