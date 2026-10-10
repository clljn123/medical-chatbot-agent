import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from embedding_utils import MyTcmEmbedding

load_dotenv()

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "tcm_knowledge"

def get_retriever(k: int = 2):
    """加载本地已有的向量库，返回检索器"""
    embeddings = MyTcmEmbedding(
        model_name="qwen3.7-text-embedding-flash",
        api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    
    vector_db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    return vector_db.as_retriever(search_kwargs={"k": k})