from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embedding
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from pinecone import Pinecone
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv(r"C:\Users\munee\anaconda_projects\16b75561-9a0c-4493-bef9-1a4554799fd9\End-to-End-Medical-Chatbot\.env")
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

embeddings = download_hugging_face_embedding()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalchatbot"

# Load existing index
docsearch = PineconeVectorStore(
    index=pc.Index(index_name),
    embedding=embeddings
)

# Prompt
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.8,
    max_tokens=512
)

# Modern LCEL chain replacing RetrievalQA
qa_chain = (
    {
        "context": docsearch.as_retriever(search_kwargs={"k": 2}) | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough()
    }
    | PROMPT
    | llm
    | StrOutputParser()
)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)
    result = qa_chain.invoke(msg)  # plain string, not dict
    print("Response:", result)
    return str(result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)