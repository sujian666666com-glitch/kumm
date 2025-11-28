import os
from dotenv import load_dotenv

# 加载环境变量, override=True 覆盖已存在的环境变量
load_dotenv(override=True)

# 配置常量
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 向量数据库配置
FAISS_DB_PATH = "faiss_db"
EMBEDDING_MODEL = "text-embedding-v1"

# 文本分割配置
CHUNK_SIZE = 100           # 每个chunk的大小
CHUNK_OVERLAP = 20         # chunk之间重叠的大小

# LLM配置
LLM_MODEL_NAME = "deepseek-chat"
LLM_MODEL_PROVIDER = "deepseek"

# 系统提示词
SYSTEM_PROMPT = """
你是AI助手，请根据提供的上下文回答问题，确保提供所有细节，如果答案不在上下文中，请说"答案不在上下文中"，不要提供错误的答案
限制：
1. 如果用户问的是法律相关问题，直接调用你的检索工具，不要根据你本身的知识回答。
2. 如果是其他问题，直接回答，不用调用工具。
"""

# 设置环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 验证必要的API密钥
def validate_api_keys():
    """
    验证必要的API密钥
    """
    missing_keys = []
    if not DEEPSEEK_API_KEY:
        missing_keys.append("DEEPSEEK_API_KEY")
    if not DASHSCOPE_API_KEY:
        missing_keys.append("DASHSCOPE_API_KEY")
    if missing_keys:
        raise ValueError(f"缺少必要的API密钥: {','.join(missing_keys)}. 请在.env文件中设置")

    # 验证目录可写
    if not os.path.exists(FAISS_DB_PATH):
        try:
            os.makedirs(FAISS_DB_PATH)
        except Exception as e:
            raise PermissionError(f"无法创建向量数据库目录：{e}")
