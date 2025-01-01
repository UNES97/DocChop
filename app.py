import os 
import chromadb
from chromadb.config import Settings
import streamlit as st 
import base64
from transformers import AutoTokenizer, AutoModelForSeq2SeqLML, pipeline
import torch 
from langchain.document_loaders import PDFMinerLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA 
from streamlit_chat import message

# ChromaDB Settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl='duckdb+parquet',
    persist_directory='db',
    anonymized_telemetry=False
)

# Initialize persistent directory
PERSIST_DIRECTORY = os.path.join(os.getcwd(), 'db')
if not os.path.exists(PERSIST_DIRECTORY):
    os.makedirs(PERSIST_DIRECTORY)

# Page configuration and styling
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-container {
        border-radius: 10px;
        padding: 20px;
        background-color: #f5f5f5;
    }
    .user-input {
        border-radius: 20px;
        border: 2px solid #4CAF50;
    }
    .file-uploader {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize device and model configurations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = "MBZUAI/LaMini-T5-738M"
PERSIST_DIR = "db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        CHECKPOINT,
        device_map=device,
        torch_dtype=torch.float32
    )
    return tokenizer, model

@st.cache_resource
def initialize_chroma_client():
    return chromadb.Client(Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
    ))

@st.cache_resource
def process_pdf(file_path):
    loader = PDFMinerLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)
    
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize ChromaDB with proper settings
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        client_settings=CHROMA_SETTINGS,
        persist_directory=PERSIST_DIRECTORY
    )
    
    # Ensure persistence
    db.persist()
    return db

@st.cache_resource
def setup_qa_chain():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize ChromaDB with existing data
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS
    )
    
    # Initialize the language model and QA chain
    tokenizer, model = load_model()
    pipe = pipeline(
        'text2text-generation',
        model=model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
        device=device
    )
    
    local_llm = HuggingFacePipeline(pipeline=pipe)
    
    return RetrievalQA.from_chain_type(
        llm=local_llm,
        chain_type="stuff",
        retriever=db.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
        ),
        return_source_documents=True
    )

def cleanup_db():
    """Cleanup function to properly close DuckDB connections"""
    try:
        client = initialize_chroma_client()
        client.reset()
    except Exception as e:
        st.error(f"Error during cleanup: {str(e)}")

def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def display_chat_history(history):
    with st.container():
        for i, (past, generated) in enumerate(zip(history["past"], history["generated"])):
            message(past, is_user=True, key=f"{i}_user")
            message(generated, key=str(i))

def main():
    st.title("📚 PDF Chat Assistant")
    st.markdown("### Upload your PDF and start chatting!")

    uploaded_file = st.file_uploader("", type=["pdf"], key="pdf_uploader")

    try:
        if uploaded_file:
            file_path = os.path.join("docs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### PDF Preview")
                display_pdf(file_path)

            with col2:
                with st.spinner('Processing PDF...'):
                    qa_chain = setup_qa_chain()
                
                st.markdown("### Chat")
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = {
                        "generated": ["Hello! How can I help you with the document?"],
                        "past": ["Hi!"]
                    }

                user_input = st.text_input("Ask a question:", key="user_input", 
                                        placeholder="Type your question here...")

                if user_input:
                    with st.spinner('Thinking...'):
                        response = qa_chain({"query": user_input})
                        st.session_state.chat_history["past"].append(user_input)
                        st.session_state.chat_history["generated"].append(response["result"])

                display_chat_history(st.session_state.chat_history)
        pass
    finally:
        cleanup_db()

import atexit
atexit.register(cleanup_db)

if __name__ == "__main__":
    main()