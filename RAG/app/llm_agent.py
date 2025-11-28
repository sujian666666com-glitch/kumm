from langchain.agents import create_agent
from langchain.tools import create_retriever_tool
from langchain_community.chat_models import ChatOpenAI
import os

# Config constants
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
LLM_MODEL_NAME = "deepseek-chat"
LLM_MODEL_PROVIDER = "https://api.deepseek.com"  # or base URL
SYSTEM_PROMPT = "You are a helpful PDF assistant that answers questions based on PDF content."

class LLMAgent:
    def __init__(self):
        """初始化代理类"""
        self.llm = self._llm_init()  # 预初始化LLM
    
    def _llm_init(self):
        """
        内部方法: 初始化LLM
        Returns:
            ChatModel: 初始化的LLM模型
        """
        if not DEEPSEEK_API_KEY:
            raise ValueError("DeepSeek API Key未设置")
        
        try:
            # DeepSeek with custom base URL
            return ChatOpenAI(
                model=LLM_MODEL_NAME,
                api_key=DEEPSEEK_API_KEY,
                base_url=LLM_MODEL_PROVIDER,
                temperature=0.3,
                timeout=30
            )
        except Exception as e:
            raise RuntimeError(f"初始化LLM大模型失败: {e}")
    
    def create_agent_executor(self, retriever):
        """
        创建agent执行器
        Args:
            retriever: 向量检索器
        Returns:
            agent: LangChain agent
        """
        # 1. 创建检索工具
        retriever_tool = create_retriever_tool(
            retriever,
            name="pdf_extractor",
            description="用于从PDF文件中检索相关信息来回答用户的问题"
        )
        tools = [retriever_tool]
        
        # 2. 创建agent（使用v1 API）
        agent = create_agent(
            model=self.llm,  # 使用之前初始化的LLM
            tools=tools,
            system_prompt=SYSTEM_PROMPT  # 使用系统提示词
        )
        
        return agent
    
    def chat(self, agent, user_question):
        """
        聊天
        Args:
            agent: agent实例
            user_question: 用户问题
        Returns:
            str: 机器人回答
        """
        if not user_question.strip():
            return "❌ 问题不能为空，请输入有效的问题"
        
        try:
            # 新API调用方式（LangChain v1）
            response = agent.invoke({
                "messages": [{"role": "user", "content": user_question.strip()}]
            })
            
            # 提取回答
            if isinstance(response, dict):
                return response.get("content", "无法获取答案，请稍后再试")
            return str(response)
        
        except Exception as e:
            return f"❌ 回答生成失败：{str(e)}"


# 使用示例
if __name__ == "__main__":
    print("⚠️  测试模式：需要先创建向量数据库才能正常问答")
    print("请运行 main.py 进行完整功能测试")