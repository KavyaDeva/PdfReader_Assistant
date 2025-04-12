# Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from numpy import e
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from scripts.document_loader import load_document
#from gemini import generate_embeddings, generate_llm
from langchain_deepseek import ChatDeepSeek
from mydeepseek import deep_embeddings, deep_llm


def load_document(pdf):
    # Load a PDF
    """
    Load a PDF and split it into chunks for efficient retrieval.

    :param pdf: PDF file to load
    :return: List of chunks of text
    """

    loader = PyPDFLoader(pdf)
    docs = loader.load()

    # Instantiate Text Splitter with Chunk Size of 500 words and Overlap of 100 words so that context is not lost
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    # Split into chunks for efficient retrieval
    chunks = text_splitter.split_documents(docs)

    print("load document")

    # Return
    return chunks


# Create a Streamlit app
st.title("AIPowered_PDF_Document_Reader")

# Load document to streamlit
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# If a file is uploaded, create the TextSplitter and vector database
if uploaded_file :

    # Code to work around document loader from Streamlit and make it readable by langchain
    temp_file = "./KavyaDeva_PM.pdf"
    with open(temp_file, "wb") as file:
        file.write(uploaded_file.getvalue())
        file_name = uploaded_file.name

    # Load document and split it into chunks for efficient retrieval.
    chunks = load_document(temp_file)

    system_prompt = (
        "You are a helpful assistant. Use the given context to answer the question."
        "If you don't know the answer, say you don't know.\n\n{context}"
    )

    # Create a prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    retriever=deep_embeddings(chunks=chunks)
    llm=deep_llm()

    # Create a chain
    # It creates a StuffDocumentsChain, which takes multiple documents (text data) and "stuffs" them together before passing them to the LLM for processing.

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    

    # Creates the RAG
    chain = create_retrieval_chain(retriever, question_answer_chain)
    print("chain created")
    print(chain)
    print("prompt")
    print(prompt)

    print(type(chunks))
    print(len(chunks))
    print(chunks[:2])  # Print first two chunks to inspect format

    # Message user that document is being processed with time emoji
st.write("Processing document... :watch:")

# Streamlit input for question
question = st.text_input("Ask a question about the document:")
if question:
        # Answer
        response = chain.invoke({"input": question})
        result = response["answer"]
        st.write(result)
        print("response")
        print(response)