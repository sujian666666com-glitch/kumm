import os
import shutil
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS

# Your config constants (make sure these are defined)
FAISS_DB_PATH = "./faiss_db"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
EMBEDDING_MODEL = "text-embedding-v1"  # Your model name

class VectorStore:
    def __init__(self):
        self.db_path = FAISS_DB_PATH
        self.embeddings = self.embeddings_model()
    
    def embeddings_model(self):
        """初始化嵌入模型"""
        if not DASHSCOPE_API_KEY:
            raise ValueError("请设置DASHSCOPE_API_KEY环境变量")
        
        return DashScopeEmbeddings(
            model=EMBEDDING_MODEL,
            dashscope_api_key=DASHSCOPE_API_KEY
        )
    
    def check_vector_database_exists(self):
        """检查向量数据库是否存在"""
        return os.path.exists(self.db_path)
    
    def create_vector_database(self, text_chunks):
        """
        创建向量数据库
        Args:
            text_chunks (list): 文本分块列表
        """
        if not text_chunks:
            raise ValueError("没有可处理的文本块")
        
        try:
            vector_store = FAISS.from_texts(
                texts=text_chunks,
                embedding=self.embeddings,
            )
            vector_store.save_local(self.db_path)
            print(f"✅ 向量数据库创建成功！保存路径：{self.db_path}")
        except Exception as e:
            raise RuntimeError(f"创建向量数据库失败：{e}")
    
    def load_vector_database(self):
        """
        加载FAISS向量数据库
        Returns:
            FAISS: 加载后的向量数据库实例
        """
        if not self.check_vector_database_exists():
            raise FileNotFoundError("向量数据库不存在, 请先处理PDF文件")
        
        try:
            return FAISS.load_local(
                self.db_path,
                embeddings=self.embeddings,
            )
        except Exception as e:
            raise RuntimeError(f"加载向量数据库失败：{e}")
    
    def delete_vector_database(self):
        """删除向量数据库"""
        if self.check_vector_database_exists():
            try:
                shutil.rmtree(self.db_path)
                print(f"✅ 向量数据库删除成功！保存路径：{self.db_path}")
                return True
            except Exception as e:
                raise RuntimeError(f"删除向量数据库失败：{e}")
        else:
            print("向量数据库不存在，无需删除!")
            return False
    
    def get_retriever(self):
        """
        获取向量检索器
        Returns:
            BaseRetriever: 向量检索器实例
        """
        if not self.check_vector_database_exists():
            raise FileNotFoundError("向量数据库不存在, 请先处理PDF文件")
        
        vector_store = self.load_vector_database()
        return vector_store.as_retriever(
            search_kwargs={"k": 3}  # 返回Top3相关chunk
        )