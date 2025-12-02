import json
import os
import sys

# ==========================================
# 修正后的导入部分 (适配最新版 LangChain)
# ==========================================
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
# 关键修改：从 langchain_core 导入 Document 和 Prompt
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ================= 配置区域 =================

# 1. Embedding 模型
EMBED_MODEL_NAME = "nomic-embed-text" 

# 2. 对话/代码模型
LLM_MODEL_NAME = "gpt-oss"

# 3. Ollama 地址
OLLAMA_URL = "http://localhost:11434"

# 4. 向量库持久化路径
VECTOR_DB_PATH = "./chroma_db_bioblend"

# ===========================================

def load_and_embed_data():
    """
    读取 bioblend_knowledge.json 并存入本地 Chroma 向量库
    """
    # 检查是否已经存在向量库
    if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
        print(f"检测到本地向量库 {VECTOR_DB_PATH} 已存在，直接加载...")
        embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME, base_url=OLLAMA_URL)
        vector_db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
        return vector_db

    print("正在初始化向量库...")
    
    json_path = "bioblend_knowledge.json"
    if not os.path.exists(json_path):
        print(f"错误：找不到 {json_path} 文件！")
        sys.exit(1)

    print(f"正在读取 {json_path} ...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        doc = Document(
            page_content=item["text_for_vector_db"], 
            metadata={
                "api_call": item["api_call"], 
                "signature": item["signature"]
            }
        )
        documents.append(doc)

    print(f"加载了 {len(documents)} 条规则。")
    print(f"正在调用本地 Ollama ({EMBED_MODEL_NAME}) 进行向量化，请稍候...")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME, base_url=OLLAMA_URL)
    
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    print("向量化完成，数据已保存到本地硬盘。")
    return vector_db

def query_local_ai(vector_db, question):
    print(f"\n>>> 正在思考问题: {question}")
    
    # 检索最相关的3条规则
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 定义本地 LLM
    llm = ChatOllama(
        model=LLM_MODEL_NAME,
        base_url=OLLAMA_URL,
        temperature=0.1 
    )

    template = """
    你是一个 Galaxy BioBlend Python 库的编程专家。
    
    请基于以下【参考文档】（API 接口说明），为用户的问题编写 Python 代码。
    
    【注意】：
    1. 必须使用 bioblend 库。
    2. 直接输出代码，不要废话。
    3. 如果需要 API Key，请用变量 'YOUR_API_KEY' 代替。

    【参考文档】：
    {context}

    【用户问题】：
    {question}
    
    【你的回答】：
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f">>> {LLM_MODEL_NAME} 回答生成中:\n")
    try:
        for chunk in chain.stream(question):
            print(chunk, end="", flush=True)
    except Exception as e:
        print(f"错误: {e}")
    print("\n")

if __name__ == "__main__":
    # 初始化
    db = load_and_embed_data()
    
    print("="*60)
    print(f"本地 Galaxy 助手已启动")
    print(f"Embedding: {EMBED_MODEL_NAME} | LLM: {LLM_MODEL_NAME}")
    print("输入 'exit' 退出")
    print("="*60)
    
    while True:
        user_input = input("\n请输入你的需求: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input.strip():
            continue
        query_local_ai(db, user_input)
