import os
# 注意：您的代码中虽然引用了 sys.path.append("..")，
# 但为了保持模块化，这里假设您会在 main.py 或运行时正确处理导入。
# 这里直接使用相对导入，假设 config 文件在 app 目录下。
from app.config import CHUNK_SIZE, CHUNK_OVERLAP 

# LangChain 导入：使用 langchain_community 和 langchain_text_splitters 保持兼容性
# 导入 PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
# 导入 TextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 导入 Document 类型
from langchain_core.documents import Document 

# 如果您坚持使用 PyPDF2 而不是 LangChain 的加载器，请导入
# from PyPDF2 import PdfReader 
# 但推荐使用 LangChain 的加载器，它通常能更好地处理文档结构和元数据。

def load_and_split_pdf(pdf_paths: list[str]) -> list[str]:
    """
    加载PDF文件，并将其分割为多个文本块。
    Args:
        pdf_paths: PDF文件路径列表（如 ["file1.pdf", "file2.pdf"]）
    Returns:
        list[str]: 分割后的文本块（chunk）列表
    """
    all_documents = []
    
    # 1. 加载PDF文档并检查文件
    print(f"📄 正在加载 {len(pdf_paths)} 个PDF文件...")
    for path in pdf_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件 {path} 不存在")
        if not path.lower().endswith(".pdf"):
            raise ValueError(f"文件 {path} 不是PDF文件")
        
        try:
            # 使用 LangChain 的 PyPDFLoader，它会自动处理文本提取
            loader = PyPDFLoader(path)
            documents = loader.load()
            all_documents.extend(documents)
        except Exception as e:
            raise RuntimeError(f"使用 PyPDFLoader 读取PDF文件失败：{path}，错误：{e}")

    if not all_documents:
        raise ValueError("所有PDF文件中均未提取到有效文档")

    # 2. 构建一个递归文本分割器
    print(f"✂️ 正在将文档分割为文本块...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        length_function = len, 
        # 优化分割逻辑：LangChain 默认的 separators 已经优化过
        separators=["\n\n", "\n", ". ", " ", ""] 
    )
    
    # split_documents 返回的是 Document 列表
    chunk_documents: list[Document] = text_splitter.split_documents(all_documents)
    
    # 将 Document 对象列表转换为纯文本列表 (因为您的 VectorStore.create_vector_database 接收的是 texts)
    text_chunks: list[str] = [doc.page_content for doc in chunk_documents]
    
    return text_chunks

# --- 测试代码 (Test Execution) ---
if __name__ == "__main__":
    # 临时配置，以便在独立运行时测试
    # 假设您的 app/config.py 已经将 CHUNK_SIZE 和 CHUNK_OVERLAP 定义为常量
    
    # ！！！请根据您实际存放的PDF路径进行修改！！！
    PDF_DOCUMENTS = ["/home/jason/workdir/llm_development/langchain_v0.3/06-文档加载器/中华人民共和国民法典.pdf"]
    
    print("#"*20 + "开始处理" + "#"*20)
    try:
        chunks = load_and_split_pdf(PDF_DOCUMENTS)
        print("#"*20 + "分割后的chunk" + "#"*20)
        print(f"数据类型: {type(chunks)}, chunk数量: {len(chunks)}")
        
        # 打印前5个chunk
        for i in range(min(5, len(chunks))):
            print(f"第{i+1}个chunk (长度: {len(chunks[i])}):\n---{chunks[i][:150]}...\n")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")