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

gi = None
print(f">>> [系统] 正在连接 Galaxy ({GALAXY_URL})...")
try:
    gi = GalaxyInstance(url=GALAXY_URL, key=GALAXY_KEY)
    user = gi.users.get_current_user()
    print(f">>> [系统] Galaxy 连接成功! 当前用户: {user.get('username', 'Unknown')}")
except Exception as e:
    print(f">>> [严重错误] Galaxy 连接失败: {e}")

class BioBlendAgent:
    def __init__(self):
        self.vector_db = self._load_db()
        self.brain = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0.1)
        self.eye = ChatOllama(model=VISION_MODEL, base_url=OLLAMA_URL, temperature=0)
        
        self.system_capabilities = """
        [Source A: Galaxy System APIs]
        1. Get Current User Info: gi.users.get_current_user()
        2. List Histories: gi.histories.get_histories()
        3. Upload File: gi.tools.upload_file('path', history_id)
        """

    def _load_db(self):
        if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
            embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
            return Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
        return None

    def ocr_image(self, image_path):
        """OCR 识别"""
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
            
            prompt_text = """
            Please analyze this image.
            1. Extract all visible text strictly.
            2. Briefly describe what this image is about in **Chinese** (Simplified).
            Output format: [识别到的文字]: ... [图片描述]: ...
            """
            message = HumanMessage(content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ])
            res = self.eye.invoke([message])
            return res.content
        except Exception as e:
            return f"OCR 识别失败: {str(e)}"

    def smart_process(self, user_query, file_context, chat_history=[], selected_tool=None):
        if not self.vector_db:
            return {"type": "text", "reply": "❌ 错误：知识库未加载。", "suggestions": []}

        if selected_tool:
            return self._generate_code_only(user_query, file_context, selected_tool)

        # RAG 流程
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(user_query)
        retrieved_tools = "\n".join([f"- Tool {i+1}: {d.page_content}" for i, d in enumerate(docs)])

        history_text = ""
        if chat_history:
            history_text = "\n".join([f"User: {h.get('user','')}\nAI: {h.get('ai','')}" for h in chat_history[-3:]])

        template = """
        You are a Galaxy BioBlend Expert. 
        【Language Rules】Follow User's Language (Chinese -> Chinese).
        【History】{history}
        【Request】User: "{query}" | File: {file_context}
        【Knowledge】{system_caps}
        [Source B] {retrieved_tools}
        
        【Decision Logic】
        1. System API -> Code.
        2. Tool Run -> Code.
        3. Missing File -> Reply "Please upload file first" in Chinese.
        4. Ambiguous -> JSON List.
        5. Chat -> Natural answer.
        
        【Output】Code: ```python ... ``` | List: ```json ... ``` | Text: Plain text.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.brain | StrOutputParser()
        
        response = chain.invoke({
            "query": user_query,
            "file_context": str(file_context),
            "system_caps": self.system_capabilities,
            "retrieved_tools": retrieved_tools,
            "history": history_text
        })
        
        return self._parse_llm_response(response, context="chat")

    def _generate_code_only(self, query, file_context, tool_info):
        template = """
        You are a Galaxy BioBlend Expert. User selected tool "{tool_name}" (ID: {tool_id}).
        User Request: "{query}" | File: {file_context}
        Task: Write Python code using `gi.tools.run_tool`. Print result.
        Return ONLY Python code inside ```python```.
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.brain | StrOutputParser()
        response = chain.invoke({
            "tool_name": tool_info['name'],
            "tool_id": tool_info['id'],
            "query": query,
            "file_context": str(file_context)
        })
        return self._parse_llm_response(response, context="execution")

    def _parse_llm_response(self, response, context="chat"):
        """解析响应并生成引导建议"""
        response = response.strip()
        suggestions = []

        # 1. 代码执行结果
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
            exec_out = self._execute_code_sandbox(code)
            final_reply = f"### 🤖 策略代码\n```python\n{code}\n```\n### ✅ 执行结果\n```text\n{exec_out}\n```"
            
            # 【新增】生成引导建议
            suggestions = [
                "能解释一下这个结果吗？",
                "如何将这些数据可视化？",
                "帮我把结果保存到本地"
            ]
            return {"type": "text", "reply": final_reply, "suggestions": suggestions}
        
        # 2. 工具选择列表
        elif "```json" in response:
            try:
                candidates = json.loads(response.split("```json")[1].split("```")[0].strip())
                return {"type": "choice", "reply": "找到多个相关工具，请选择：", "candidates": candidates, "suggestions": []}
            except:
                return {"type": "text", "reply": response, "suggestions": []}
        
        # 3. 纯文本 (可能是缺文件提示，或者闲聊)
        else:
            # 简单的关键词匹配来生成建议
            if "上传" in response or "upload" in response.lower():
                suggestions = ["如何获取示例数据？", "支持哪些文件格式？"]
            elif "历史" in response:
                suggestions = ["列出最近的 dataset", "清理历史记录"]
            
            return {"type": "text", "reply": response, "suggestions": suggestions}

    def _execute_code_sandbox(self, code):
        if not gi: return "Galaxy 未连接"
        output_buffer = io.StringIO()
        try:
            sandbox = {"gi": gi, "json": json, "print": print}
            with contextlib.redirect_stdout(output_buffer):
                exec(code, sandbox)
            return output_buffer.getvalue() or "(无输出)"
        except Exception:
            return traceback.format_exc()

app = FastAPI()
templates = Jinja2Templates(directory="templates")
agent = BioBlendAgent()

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = [] 
    selected_tool: Optional[Dict[str, Any]] = None
    uploaded_file_id: Optional[str] = None
    uploaded_file_name: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/upload")
async def upload_handler(file: UploadFile = File(...)):
    try:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        result = {"status": "success", "file_name": file.filename}
        
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            ocr_text = agent.ocr_image(temp_path)
            result["type"] = "image"
            result["ocr_text"] = ocr_text
            # 图片识别后的建议
            result["suggestions"] = ["提取其中的序列信息", "解释报错原因", "翻译成中文"]
            os.remove(temp_path) 
        else:
            if gi:
                histories = gi.histories.get_histories()
                hid = histories[0]['id'] if histories else gi.histories.create_history("GPT-OSS Analysis")['id']
                ret = gi.tools.upload_file(temp_path, hid)
                result["type"] = "data"
                result["file_id"] = ret['outputs'][0]['id']
                # 数据上传后的建议
                result["suggestions"] = ["对这个文件做质控", "查看文件前10行", "比对到参考基因组"]
                os.remove(temp_path)
            else:
                return {"status": "error", "message": "Galaxy 未连接"}
        return result
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.post("/api/chat")
async def chat_handler(req: ChatRequest):
    file_ctx = {
        "has_file": bool(req.uploaded_file_id),
        "file_id": req.uploaded_file_id,
        "file_name": req.uploaded_file_name
    }
    response = agent.smart_process(req.message, file_ctx, req.history, req.selected_tool)
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
