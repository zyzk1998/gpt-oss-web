import os
import io
import json
import base64
import traceback
import contextlib
import uvicorn
import shutil
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
from dotenv import load_dotenv

# --- LangChain & BioBlend ---
from bioblend.galaxy import GalaxyInstance
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

# 1. 加载环境变量
load_dotenv()

# ================= 配置区域 =================
PORT = 8082
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "gpt-oss:latest"
VISION_MODEL = "llama3.2-vision:11b"
OLLAMA_URL = "http://localhost:11434"
VECTOR_DB_PATH = "./data/chroma_db_bioblend"

GALAXY_URL = os.getenv("GALAXY_URL", "https://usegalaxy.org")
GALAXY_KEY = os.getenv("GALAXY_API_KEY", "")
# ===========================================

# --- 2. 初始化 Galaxy 连接 ---
gi = None
print(f">>> [系统] 正在连接 Galaxy ({GALAXY_URL})...")
try:
    gi = GalaxyInstance(url=GALAXY_URL, key=GALAXY_KEY)
    user = gi.users.get_current_user()
    print(f">>> [系统] Galaxy 连接成功! 当前用户: {user.get('username', 'Unknown')}")
except Exception as e:
    print(f">>> [严重错误] Galaxy 连接失败: {e}")

# --- 3. 全能智能体定义 ---
class BioBlendAgent:
    def __init__(self):
        self.vector_db = self._load_db()
        # 大脑: 负责推理和代码生成 (temperature=0.1 保证逻辑稳定)
        self.brain = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0.1)
        # 眼睛: 负责看图
        self.eye = ChatOllama(model=VISION_MODEL, base_url=OLLAMA_URL, temperature=0)
        
        # 系统基础能力 (Source A)
        self.system_capabilities = """
        [Source A: Galaxy System APIs]
        1. Get Current User Info: gi.users.get_current_user()
        2. List Histories: gi.histories.get_histories()
        3. Upload File: gi.tools.upload_file('path', history_id)
        """

    def _load_db(self):
        if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
            print(f">>> [系统] 加载向量库: {VECTOR_DB_PATH}")
            embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
            return Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
        print(">>> [警告] 向量库未找到，请先运行 rebuild_db.py")
        return None

    def ocr_image(self, image_path):
        """利用视觉模型识别图片内容 (强制中文输出)"""
        print(f"   [视觉] 正在调用 {VISION_MODEL} 进行 OCR...")
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
            
            # 【优化】Prompt: 要求提取文本并用中文描述
            prompt_text = """
            Please analyze this image.
            1. Extract all visible text strictly.
            2. Briefly describe what this image is about in **Chinese** (Simplified).
            
            Output format:
            [识别到的文字]: ...
            [图片描述]: ...
            """
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]
            )
            res = self.eye.invoke([message])
            return res.content
        except Exception as e:
            print(f"   [视觉错误] {e}")
            return f"OCR 识别失败: {str(e)}"

    def smart_process(self, user_query, file_context, chat_history=[], selected_tool=None):
        """
        核心逻辑：RAG + 记忆 + 决策
        """
        if not self.vector_db:
            return {"type": "text", "reply": "❌ 错误：知识库未加载，无法工作。"}

        # A. 用户已选定工具 -> 强制生成代码
        if selected_tool:
            return self._generate_code_only(user_query, file_context, selected_tool)

        # B. 常规流程
        print(f"   [思考] 用户需求: {user_query}")
        
        # 1. 检索 (Source B)
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(user_query)
        retrieved_tools = "\n".join([f"- Tool {i+1}: {d.page_content}" for i, d in enumerate(docs)])

        # 2. 格式化历史记录 (只取最近 3 轮，避免 Token 溢出)
        history_text = ""
        if chat_history:
            history_text = "\n".join([f"User: {h.get('user','')}\nAI: {h.get('ai','')}" for h in chat_history[-3:]])

        # 3. 构造 Prompt (增加语言约束和记忆槽)
        template = """
        You are a Galaxy BioBlend Expert. 
        
        【Language Rules】
        1. **Follow User's Language**: If the user asks in Chinese, you MUST reply in Chinese. If English, reply in English.
        2. **Exception**: Do NOT translate the Python code or Galaxy Tool Names.
        
        【Conversation History】
        {history}
        
        【Current Request】
        User: "{query}"
        File Status: {file_context}
        
        【Knowledge Base】
        {system_caps}
        
        [Source B: Retrieved Tools]
        {retrieved_tools}
        
        【Decision Logic】
        1. **System API**: If it matches Source A (e.g., "who am I"), generate code.
        2. **Tool Run**: If it matches Source B (e.g., "run FastQC"), generate code.
        3. **Missing File**: If tool needs file but File Status is empty, reply in Chinese: "请先上传文件 (Please upload file first)."
        4. **Ambiguous**: If unsure, return a JSON list.
        5. **Chat**: If the user is just chatting or asking about previous results (based on History), answer them naturally.
        
        【Output Format】
        - Code: ```python ... ```
        - List: ```json ... ```
        - Text: Plain text (in User's Language).
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.brain | StrOutputParser()
        
        print("   [推理] gpt-oss 正在决策...")
        response = chain.invoke({
            "query": user_query,
            "file_context": str(file_context),
            "system_caps": self.system_capabilities,
            "retrieved_tools": retrieved_tools,
            "history": history_text
        })
        
        return self._parse_llm_response(response)

    def _generate_code_only(self, query, file_context, tool_info):
        """强制生成代码 (用于用户选定工具后)"""
        template = """
        You are a Galaxy BioBlend Expert. User selected tool "{tool_name}" (ID: {tool_id}).
        User Request: "{query}"
        File Status: {file_context}
        
        Task: Write Python code using `gi.tools.run_tool`.
        Requirements:
        1. Assume `gi` is connected.
        2. Use file ID from status if available.
        3. MUST print result.
        4. Return ONLY Python code inside ```python```.
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.brain | StrOutputParser()
        response = chain.invoke({
            "tool_name": tool_info['name'],
            "tool_id": tool_info['id'],
            "query": query,
            "file_context": str(file_context)
        })
        return self._parse_llm_response(response)

    def _parse_llm_response(self, response):
        """解析 LLM 返回的混合格式"""
        response = response.strip()
        
        # 1. 识别代码块 -> 执行
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
            exec_out = self._execute_code_sandbox(code)
            final_reply = f"### 🤖 策略代码\n```python\n{code}\n```\n### ✅ 执行结果\n```text\n{exec_out}\n```"
            return {"type": "text", "reply": final_reply}
        
        # 2. 识别 JSON (工具列表) -> 前端选择
        elif "```json" in response:
            try:
                json_str = response.split("```json")[1].split("```")[0].strip()
                candidates = json.loads(json_str)
                return {
                    "type": "choice",
                    "reply": "找到多个相关工具，请选择：",
                    "candidates": candidates
                }
            except:
                return {"type": "text", "reply": response}
        
        # 3. 尝试直接解析 JSON
        elif response.startswith("[") and response.endswith("]"):
            try:
                candidates = json.loads(response)
                return {"type": "choice", "reply": "请选择工具：", "candidates": candidates}
            except:
                pass

        # 4. 默认文本
        return {"type": "text", "reply": response}

    def _execute_code_sandbox(self, code):
        """沙箱执行代码"""
        if not gi: return "Galaxy 未连接，无法执行。"
        
        output_buffer = io.StringIO()
        try:
            # 注入必要的全局变量
            sandbox = {"gi": gi, "json": json, "print": print}
            with contextlib.redirect_stdout(output_buffer):
                exec(code, sandbox)
            result = output_buffer.getvalue()
            return result if result else "(代码执行成功，但没有 print 输出)"
        except Exception:
            return traceback.format_exc()

# --- 4. Web 应用初始化 ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")
agent = BioBlendAgent()

# 【优化】请求模型增加 history 字段
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = [] 
    selected_tool: Optional[Dict[str, Any]] = None
    uploaded_file_id: Optional[str] = None
    uploaded_file_name: Optional[str] = None

# --- 路由定义 ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/upload")
async def upload_handler(file: UploadFile = File(...)):
    """处理文件上传：图片->OCR，数据->Galaxy"""
    try:
        # 保存临时文件
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        result = {"status": "success", "file_name": file.filename}
        
        # 分支 A: 图片 (OCR)
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            print(f">>> [上传] 检测到图片，启动 Vision 模型...")
            ocr_text = agent.ocr_image(temp_path)
            result["type"] = "image"
            result["ocr_text"] = ocr_text
            os.remove(temp_path) 
            
        # 分支 B: 数据文件 (上传 Galaxy)
        else:
            if gi:
                print(f">>> [上传] 上传数据到 Galaxy...")
                histories = gi.histories.get_histories()
                hid = histories[0]['id'] if histories else gi.histories.create_history("GPT-OSS Analysis")['id']
                
                ret = gi.tools.upload_file(temp_path, hid)
                result["type"] = "data"
                result["file_id"] = ret['outputs'][0]['id']
                os.remove(temp_path)
            else:
                return {"status": "error", "message": "Galaxy 未连接"}
                
        return result

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.post("/api/chat")
async def chat_handler(req: ChatRequest):
    # 构建上下文
    file_ctx = {
        "has_file": bool(req.uploaded_file_id),
        "file_id": req.uploaded_file_id,
        "file_name": req.uploaded_file_name
    }
    
    # 统一入口，传入 history
    response = agent.smart_process(req.message, file_ctx, req.history, req.selected_tool)
    return response

if __name__ == "__main__":
    print(f">>> [启动] 服务运行在: http://0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
