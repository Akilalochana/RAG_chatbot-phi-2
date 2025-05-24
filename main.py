# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

app = FastAPI()

# Load model and components (same as Colab)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
device = 0 if torch.cuda.is_available() else -1
hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128,
                       do_sample=True, temperature=0.7, top_p=0.9,
                       pad_token_id=tokenizer.eos_token_id,
                       return_full_text=False, device=device)
llm = HuggingFacePipeline(pipeline=hf_pipeline)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
output_parser = StrOutputParser()

# Load PDF and prepare Chroma vectorstore
loader = PyPDFLoader("hotelwinwin_details.pdf")  # Replace with actual PDF path
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60,
                                               separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""])
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

template = """You are Hotel Win Win's assistant. Answer only hotel-related questions using the context below.

Context: {context}

Question: {question}

If the question is not about the hotel, respond: "I can only assist with Hotel Win Win inquiries."

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    context = "\n\n".join(doc.page_content for doc in docs)
    return context[:1200] + "..." if len(context) > 1200 else context

def clean_answer(answer):
    for stop_token in ["User:", "Question:", "Assistant:"]:
        if stop_token in answer:
            answer = answer.split(stop_token)[0]
    return answer.strip().split("\n")[0].strip()

def get_user_friendly_response(question):
    docs = retriever.get_relevant_documents(question)
    context = format_docs(docs)
    if not context or context.strip() == "":
        return "I don't know."
    user_prompt = prompt.format(context=context, question=question)
    raw_response = llm.invoke(user_prompt)
    response = clean_answer(raw_response)
    if ("i don't know" in response.lower()) or (not response) or (len(response) < 5):
        return "I don't know."
    return response

# Define input model
class Query(BaseModel):
    question: str

# API route
@app.post("/chat")
def chat(query: Query):
    answer = get_user_friendly_response(query.question)
    return {"response": answer}
