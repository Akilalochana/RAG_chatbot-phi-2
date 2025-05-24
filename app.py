from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Use smaller, faster models
llm_model_name = "google/flan-t5-small"
embedding_model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
cache_dir = "/tmp"

tokenizer = AutoTokenizer.from_pretrained(llm_model_name, cache_dir=cache_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(
    llm_model_name,
    cache_dir=cache_dir
)
device = 0 if torch.cuda.is_available() else -1

hf_pipeline = pipeline(
    "text2text-generation",
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

from langchain.embeddings import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    cache_folder=cache_dir
)

from langchain.schema.output_parser import StrOutputParser
output_parser = StrOutputParser()

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("hotelwinwin_details.pdf")
docs = loader.load()
print(f"Number of pages: {len(docs)}")

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=60,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)
splits = text_splitter.split_documents(docs)
print(f"Number of chunks: {len(splits)}")

from langchain_chroma import Chroma
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

from langchain.prompts import ChatPromptTemplate

template = """You are Hotel Win Win's assistant. Answer only hotel-related questions using the context below.

Context: {context}

Question: {question}

If the question is not about the hotel, respond: "I can only assist with Hotel Win Win inquiries."

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
    # If context is empty or just whitespace, don't answer
    if not context or context.strip() == "":
        return "I don't know."
    user_prompt = prompt.format(context=context, question=question)
    raw_response = llm.invoke(user_prompt)
    response = clean_answer(raw_response)
    # If the model "hallucinates" or says something generic, force "I don't know."
    if ("i don't know" in response.lower()) or (not response) or (len(response) < 5):
        return "I don't know."
    # Optional: If the answer is not supported by context, you can tune this filter further.
    return response

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    answer = get_user_friendly_response(query.question)
    return {"answer": answer}