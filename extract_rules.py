import inspect
import json
from bioblend.galaxy import GalaxyInstance

# 1. 初始化一个假的 Galaxy 实例，仅用于获取结构
# 不需要真实的 URL 或 Key，因为我们要的是代码结构
gi = GalaxyInstance("http://localhost", "fake_key")

# 2. 定义我们要提取的子模块 (BioBlend 将功能分成了这些类)
sub_clients = [
    'tools', 'histories', 'users', 'libraries',
    'datasets', 'workflows', 'invocations', 'jobs'
]

knowledge_base = []

for client_name in sub_clients:
    # 获取子客户端对象，例如 gi.tools
    client_obj = getattr(gi, client_name)

    # 获取该对象的所有成员方法
    for name, func in inspect.getmembers(client_obj, predicate=inspect.ismethod):
        # 过滤掉内部方法（以_开头）
        if name.startswith('_'):
            continue

 # 3. 提取函数签名（参数列表）
        try:
            signature = str(inspect.signature(func))
        except ValueError:
            signature = "(...)"

        # 4. 提取文档字符串（这是核心，告诉 LLM 这个函数怎么用）
        docstring = inspect.getdoc(func) or "无文档说明"

        # 5. 组装成一条“知识”
        # 格式：模块.方法名(参数)
        full_function_name = f"gi.{client_name}.{name}"

        entry = {
            "api_call": full_function_name,
            "signature": signature,
            "description": docstring,
            # 组合一个用于向量化的文本块
            "text_for_vector_db": f"""
            功能名称: {full_function_name}
            调用方式: {full_function_name}{signature}
            功能描述: {docstring}
            """
        }

 knowledge_base.append(entry)

# 6. 保存为 JSON，你可以直接拿这个文件去建立向量库
with open("bioblend_knowledge.json", "w", encoding='utf-8') as f:
    json.dump(knowledge_base, f, ensure_ascii=False, indent=2)

print(f"成功提取了 {len(knowledge_base)} 个 BioBlend API 方法！")
