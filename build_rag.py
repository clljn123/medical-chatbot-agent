import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from embedding_utils import MyTcmEmbedding

load_dotenv()
DOC_FOLDER = "./data/docs"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "tcm_knowledge"

def build_vectorstore():
    print("===== 开始加载知识库文档 =====")
    all_docs = []
    for filename in os.listdir(DOC_FOLDER):
        if filename.endswith(".md"):
            loader = TextLoader(os.path.join(DOC_FOLDER, filename), encoding="utf-8")
            all_docs.extend(loader.load())
            print(f"成功加载: {filename}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", "。", " ", ""])
    chunks = splitter.split_documents(all_docs)
    
    embeddings = MyTcmEmbedding(
        model_name="qwen3.7-text-embedding-flash",
        api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    
    print(f"切分后 chunk 数量: {len(chunks)}，开始向量化...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )
    print("向量库构建完成！")

if __name__ == "__main__":
    build_vectorstore()