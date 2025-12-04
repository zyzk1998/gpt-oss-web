import os
import io
import json
import sys
import contextlib
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# --- 1. 核心库导入 (RAG + Galaxy) ---
from bioblend.galaxy import GalaxyInstance
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 加载配置
load_dotenv()

# ================= 配置区域 =================
PORT = 8082
EMBED_MODEL = "nomic-embed-text"  # 确保 Ollama 已拉取此模型
LLM_MODEL = "gpt-oss"             # 你的大模型名称
OLLAMA_URL = "http://localhost:11434"
VECTOR_DB_PATH = "./data/chroma_db_bioblend"
# ===========================================

# --- 2. 定义 RAG 智能体 (带查询优化功能) ---
class BioBlendAgent:
    def __init__(self):
        self.vector_db = self._load_db()
        self.llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0.1)

    def _load_db(self):
        """加载本地向量库"""
        if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
            print(">>> [系统] 加载本地向量知识库...")
            embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
            return Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
        print(">>> [警告] 向量库未找到，请先运行 rebuild_db.py")
        return None

    def _optimize_query(self, user_question):
        """
        【核心升级】查询扩展：把用户的中文/模糊提问，转化为 Galaxy 工具的英文关键词
        策略：动作(Action) + 对象(Object) + 潜在工具名(Common Tools)
        """
        print(f"   [思考] 正在对用户问题进行'三维关键词'扩展...")
        
        template = """
        你是一个生物信息学搜索专家。用户的需求是："{question}"
        
        请生成用于在 Galaxy 工具库中检索的英文关键词。为了保证查全率，请从以下三个维度生成：
        1. 【动作】(Action): 如 Align, Map, Filter, Plot, Convert, Get
        2. 【对象】(Object): 如 FASTQ, BAM, VCF, Tabular, User Info, History
        3. 【潜在工具名】(Potential Tools): 列出该领域最著名的工具名（如 BWA, Bowtie, Samtools, DESeq2），即使不确定服务器是否安装。
        
        规则：
        - 必须转换为英文。
        - 不要写句子，只返回空格分隔的单词列表。
        - 包含 bioblend API 关键词（如 get_current_user）如果涉及系统操作。
        
        关键词列表：
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        # 获取扩展关键词
        generated_keywords = chain.invoke({"question": user_question}).strip()
        
        # 将“用户原话”和“生成的关键词”拼在一起去搜
        final_query = f"{user_question} {generated_keywords}"
        
        print(f"   [扩展] 原问: '{user_question}'")
        print(f"   [生成] 扩充: '{generated_keywords}'")
        print(f"   [最终检索词] -> '{final_query}'")
        
        return final_query

    def generate_code(self, question):
        if not self.vector_db: return "print('错误：知识库未加载')"

        # 1. 优化查询 (Query Expansion)
        search_query = self._optimize_query(question)

        # 2. 检索 (扩大范围 k=10，防止漏网之鱼)
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 10})
        docs = retriever.invoke(search_query)
        
        # (调试用) 打印检索到的第一条规则标题，看看准不准
        if docs:
            print(f"   [检索] 命中 {len(docs)} 条规则。Top1: {docs[0].metadata.get('api_call')}")

        # 3. 生成代码
        template = """
        你是一个 Galaxy BioBlend 专家。
        
        【任务】：根据用户需求和参考工具，生成 Python 代码。
        【用户需求】：{question}
        
        【参考工具规则】：
        {context}

        【代码要求】：
        1. 假设变量 `gi` (GalaxyInstance) 已连接，直接使用。
        2. 不要写 `if __name__ == "__main__":`。
        3. 直接调用工具/API，不要写伪代码。
        4. 必须 print 结果，否则前端看不到输出。
        5. 只返回纯 Python 代码，不要 Markdown 标记。
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = ({"context": lambda x: docs, "question": RunnablePassthrough()}
                 | prompt | self.llm | StrOutputParser())

        code = chain.invoke(question)
        
        # 清洗代码
        return code.replace("```python", "").replace("```", "").strip()

# --- 3. 初始化 Web 应用 ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 初始化资源
agent = BioBlendAgent()
GALAXY_URL = os.getenv("GALAXY_URL", "https://usegalaxy.org")
GALAXY_KEY = os.getenv("GALAXY_API_KEY", "")

print(f">>> [系统] 正在连接 Galaxy ({GALAXY_URL})...")
try:
    gi = GalaxyInstance(url=GALAXY_URL, key=GALAXY_KEY)
    # 尝试获取用户信息以验证连接
    try:
        user = gi.users.get_current_user()
        print(f">>> [系统] Galaxy 连接成功: {user.get('username', 'Unknown')}")
    except Exception:
        print(f">>> [系统] Galaxy 连接成功 (但无法获取用户信息，可能是 Key 权限问题)")
except Exception as e:
    print(f">>> [警告] Galaxy 连接失败: {e}")
    gi = None

# --- 4. 路由定义 ---

# 页面路由
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class ChatRequest(BaseModel):
    message: str

# 接口路由
@app.post("/api/chat")
async def chat(req: ChatRequest):
    print(f"\n>>> 收到指令: {req.message}")

    # 1. 生成代码
    code = agent.generate_code(req.message)

    # 2. 执行代码
    output_buffer = io.StringIO()
    exec_result = ""

    if gi:
        try:
            print("   [执行] 正在沙箱中运行代码...")
            with contextlib.redirect_stdout(output_buffer):
                # 注入 gi 和 json 供代码使用
                exec(code, {}, {"gi": gi, "json": json})
            exec_result = output_buffer.getvalue()
            if not exec_result:
                exec_result = "(代码执行成功，但没有 print 输出)"
        except Exception as e:
            exec_result = f"执行报错: {e}"
    else:
        exec_result = "Galaxy 未连接，仅生成代码，未执行。"

    # 3. 格式化返回
    reply = f"**生成的策略代码：**\n```python\n{code}\n```\n\n**执行结果：**\n```text\n{exec_result}\n```"
    return {"reply": reply}

if __name__ == "__main__":
    print(f">>> [启动] 请在浏览器访问: http://localhost:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
