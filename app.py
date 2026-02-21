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
import logging

# Setup logging FIRST
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("Starting app.py")
logger.debug(f"Current working dir: {os.getcwd()}")
logger.debug(f"Files in current dir: {os.listdir('.')}")

# Define Flask app EARLY (before any route decorators)
app = Flask(__name__)

# Load env vars
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set in environment variables!")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in environment variables!")

logger.debug("API keys loaded successfully")

# Heavy initialization (embeddings, vector store, LLM, chain)
embeddings = download_hugging_face_embedding()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalchatbot"

docsearch = PineconeVectorStore(
    index=pc.Index(index_name),
    embedding=embeddings
)

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.8,
    max_tokens=512
)

qa_chain = (
    {
        "context": docsearch.as_retriever(search_kwargs={"k": 2}) | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough()
    }
    | PROMPT
    | llm
    | StrOutputParser()
)

# Routes (now app is defined)
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)
    
    # Handle greetings separately
    greetings = ["hello", "hi", "hey", "how are you", "good morning", "good evening"]
    if msg.lower().strip() in greetings:
        return "Hello! I'm your Medical Assistant. Ask me any medical questions and I'll help you!"
    
    result = qa_chain.invoke(msg)
    print("Response:", result)
    return str(result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
