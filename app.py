from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

# Load LLM and supporting objects (do this ONCE)
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
device = 0 if torch.cuda.is_available() else -1
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,
    return_full_text=False,
    device=device
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

loader = PyPDFLoader("hotelwinwin_details.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=60,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

template = """You are a helpful assistant in hotel win win. Answer the following question using only the provided context. Please consider the question and provide an answer. you are the intelligent, professional bot.If you feel like what you're asking is irrelevant in context, "say i don't know""

Context:
{context}

Question: {question}
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    context = "\n\n".join(doc.page_content for doc in docs)
    # Limit context length
    if len(context) > 1200:
        context = context[:1200] + "..."
    return context.strip()

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

# FastAPI App
app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    answer = get_user_friendly_response(query.question)
    return {"answer": answer}