from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
import os

# Initialize FastAPI
app = FastAPI()

# Enable CORS for all origins (change "*" to specific domain in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request body structure
class Query(BaseModel):
    question: str

# Load PDF and prepare vector store (this runs on startup)
print("📄 Loading and indexing PDF...")
pdf_path = "hotelwinwin_details.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

persist_directory = "chroma_db"
vectordb = Chroma.from_documents(documents, embedding=embeddings, persist_directory=persist_directory)
vectordb.persist()
retriever = vectordb.as_retriever()

# Initialize LLM
print("🤖 Loading TinyLlama model...")
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.1
)

llm = HuggingFacePipeline(pipeline=pipe)

# Combine LLM + Retriever into RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# Endpoint for chatbot
@app.post("/chat")
async def chat(query: Query):
    response = qa_chain.run(query.question)
    return {"answer": response}
