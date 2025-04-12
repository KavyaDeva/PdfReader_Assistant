# Imports
from  langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_deepseek import ChatDeepSeek
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from API import DEEPSEEK_API_KEY

def deep_embeddings(chunks):
    print("ksjbagncfkshfgkg")
    # Generate embeddings
    # Embeddings are numerical vector representations of data, typically used to capture relationships, similarities,
    # and meanings in a way that machines can understand. They are widely used in Natural Language Processing (NLP),
    # recommender systems, and search engines.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Can also use HuggingFaceEmbeddings
    # from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create vector database containing chunks and embeddings
    vector_db = FAISS.from_documents(chunks, embeddings)

    # Create a document retriever
    retriever = vector_db.as_retriever()
    return retriever


def deep_llm():
    os.environ['DEEPSEEK_API_KEY'] = st.secrets['DEEPSEEK_API_KEY']
    key=os.environ['DEEPSEEK_API_KEY']
    print("dfrvbghnmjk")
    llm = ChatDeepSeek(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    temperature=0,
    api_key=key)
    return llm
