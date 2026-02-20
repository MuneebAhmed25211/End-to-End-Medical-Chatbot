from src.helper import load_pdf, text_split, download_hugging_face_embedding
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv(r"C:\Users\munee\anaconda_projects\16b75561-9a0c-4493-bef9-1a4554799fd9\End-to-End-Medical-Chatbot\.env")

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embedding()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalchatbot"

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

docsearch = PineconeVectorStore.from_texts(
    [t.page_content for t in text_chunks],
    embeddings,
    index_name=index_name,
    pinecone_api_key=PINECONE_API_KEY
)