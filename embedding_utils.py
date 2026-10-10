import dashscope
from langchain_core.embeddings import Embeddings
from typing import List

class MyTcmEmbedding(Embeddings):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        dashscope.api_key = self.api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档，内置分批处理"""
        batch_size = 10  # DashScope 单次最多处理 10 条
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                resp = dashscope.TextEmbedding.call(
                    model=self.model_name,
                    input=batch,
                    text_type="document"
                )
                if resp.status_code == 200:
                    all_embeddings.extend([item["embedding"] for item in resp.output["embeddings"]])
                else:
                    raise Exception(f"嵌入文档失败：{resp.code} - {resp.message}")
            except Exception as e:
                print(f"调用嵌入API时出错（批次 {i}）：{e}")
                raise
                
        return all_embeddings

    def embed_query(self, text) -> List[float]:
        # 兼容 LangChain 可能传入的消息对象
        if hasattr(text, 'content'):
            text = text.content
        if not isinstance(text, str):
            text = str(text)
        
        try:
            resp = dashscope.TextEmbedding.call(
                model=self.model_name,
                input=[text],  # 确保是纯字符串列表
                text_type="query"
            )
            if resp.status_code == 200:
                return resp.output["embeddings"][0]["embedding"]
            else:
                raise Exception(f"嵌入查询失败：{resp.code} - {resp.message}")
        except Exception as e:
            print(f"调用查询嵌入API时出错：{e}")
            raise
