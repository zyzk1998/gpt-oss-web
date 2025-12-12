import requests
import time
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import os

# ================= 配置区域 =================
API_URL = "http://localhost:8082/api/chat"
TOTAL_REQUESTS_PER_CATEGORY = 20  # ⚠️ 建议先设为 20 进行测试，确认无误后再改为 100
CONCURRENCY = 1  # ⚠️ 强烈建议设为 1。32B 模型显存占用高，并发会导致 OOM 或极慢
OUTPUT_DIR = "./benchmark_reports"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 测试数据集 (Prompt Pools) =================
PROMPT_POOLS = {
    "Biomedical_QA": [
        "解释一下 P53 基因在癌症中的作用",
        "什么是中心法则（Central Dogma）？",
        "CRISPR-Cas9 的工作原理是什么？",
        "线粒体为什么被称为细胞的动力工厂？",
        "解释单细胞测序中的 Dropout 现象",
        "什么是 T 细胞耗竭？",
        "DNA 甲基化如何影响基因表达？",
        "介绍一下阿尔茨海默病的病理机制",
        "什么是 GWAS 研究？",
        "RNA-seq 和 scRNA-seq 的主要区别是什么？"
    ],
    "Bioinfo_Concept": [
        "如何使用 Seurat 进行数据归一化？",
        "FastQC 报告中的 GC Content 异常说明什么？",
        "解释 PCA 降维在生信分析中的意义",
        "如何过滤单细胞数据中的双细胞（Doublets）？",
        "DESeq2 和 edgeR 有什么区别？",
        "什么是 Batch Effect（批次效应），如何去除？",
        "如何解读火山图（Volcano Plot）？",
        "Bam 文件和 Sam 文件有什么区别？",
        "什么是 UMAP？它和 t-SNE 有什么不同？",
        "如何进行 GO 富集分析？"
    ],
    "General_Chat": [
        "你好，介绍一下你自己",
        "给我讲个笑话",
        "今天天气怎么样？",
        "写一首关于 DNA 的诗",
        "你是谁开发的？",
        "1+1等于几？",
        "帮我写一封给导师的邮件草稿",
        "推荐几本好书",
        "什么是人工智能？",
        "翻译这句话成英文：生信分析很有趣"
    ],
    "Galaxy_Intent": [
        "列出所有 Seurat 相关的工具",          # 期望: choice 或 text
        "我要做单细胞分析，请规划工作流",       # 期望: workflow_config
        "Run Seurat Create Object",           # 期望: tool_config
        "帮我查找过滤细胞的工具",              # 期望: choice 或 tool_config
        "执行 Seurat 归一化",                 # 期望: tool_config
        "Show me Seurat tools",
        "规划一个 Seurat 流程",               # 期望: workflow_config
        "Run PCA",                            # 期望: tool_config
        "Find Neighbors tool",
        "Run UMAP visualization"
    ]
}

# ================= 核心测试逻辑 =================

def send_request(category, prompt, req_id):
    payload = {
        "message": prompt,
        "history": [],
        "uploaded_files": []
    }
    
    start_time = time.time()
    result = {
        "id": req_id,
        "category": category,
        "prompt": prompt,
        "status_code": 0,
        "latency": 0,
        "response_type": "error",
        "thought_len": 0,
        "success": False,
        "error_msg": ""
    }

    try:
        # 设置较长的超时时间，因为 32B 模型推理较慢
        response = requests.post(API_URL, json=payload, timeout=600) 
        end_time = time.time()
        
        result["latency"] = round(end_time - start_time, 2)
        result["status_code"] = response.status_code
        
        if response.status_code == 200:
            data = response.json()
            result["response_type"] = data.get("type", "unknown")
            # 统计思考过程的长度（字符数），反映推理深度
            thought_content = data.get("thought", "")
            result["thought_len"] = len(thought_content) if thought_content else 0
            
            # === 成功判定逻辑 ===
            if category == "Galaxy_Intent":
                # 对于工具/流程意图，只要返回了结构化配置或选择列表，就算成功
                if result["response_type"] in ["tool_config", "workflow_config", "choice", "data_selector"]:
                    result["success"] = True
                # 如果是“列出”，返回 text 但包含列表内容也算对
                elif result["response_type"] == "text" and ("1." in data.get("reply", "") or "-" in data.get("reply", "")):
                    result["success"] = True
                else:
                    result["success"] = False
            else:
                # 对于问答类，只要返回 text 且不为空就算成功
                if result["response_type"] == "text" and len(data.get("reply", "")) > 10:
                    result["success"] = True
                else:
                    result["success"] = False
        else:
            result["error_msg"] = f"HTTP {response.status_code}"
            
    except Exception as e:
        result["error_msg"] = str(e)
        result["latency"] = round(time.time() - start_time, 2)
    
    return result

def run_benchmark():
    results = []
    total_tasks = len(PROMPT_POOLS) * TOTAL_REQUESTS_PER_CATEGORY
    
    print(f"\n🚀 [GIBH Qwen Galaxy] 性能基准测试启动")
    print(f"🧠 模型: {LLM_MODEL} (32B)")
    print(f"📊 总任务数: {total_tasks} (每类 {TOTAL_REQUESTS_PER_CATEGORY} 次)")
    print(f"🖥️  并发线程: {CONCURRENCY}")
    print("-" * 50)
    
    # 准备任务队列
    tasks = []
    req_id = 0
    for category, prompts in PROMPT_POOLS.items():
        for _ in range(TOTAL_REQUESTS_PER_CATEGORY):
            prompt = random.choice(prompts)
            tasks.append((category, prompt, req_id))
            req_id += 1
            
    # 进度条
    pbar = tqdm(total=total_tasks, desc="Testing", unit="req")
    
    # 线程池执行
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        future_to_req = {executor.submit(send_request, t[0], t[1], t[2]): t for t in tasks}
        
        for future in as_completed(future_to_req):
            res = future.result()
            results.append(res)
            pbar.update(1)
            
            # 实时错误日志
            if not res["success"] and res["status_code"] == 200:
                # 意图识别失败（比如问工具却回了闲聊）
                tqdm.write(f"⚠️ [Intent Fail] {res['category']} -> Got {res['response_type']} | Prompt: {res['prompt'][:15]}...")
            elif res["status_code"] != 200:
                # 系统错误
                tqdm.write(f"❌ [Sys Error] {res['error_msg']}")

    pbar.close()
    return pd.DataFrame(results)

# ================= 可视化报告生成 =================

def generate_report(df):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存 CSV
    csv_path = f"{OUTPUT_DIR}/benchmark_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 原始数据已保存: {csv_path}")

    # 2. 设置绘图风格
    sns.set_theme(style="whitegrid")
    # 尝试设置中文字体，如果没有则回退到英文
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'GIBH Qwen Galaxy (32B) Performance Benchmark', fontsize=20, y=0.95)

    # --- 图 1: 平均延迟 (Bar Plot) ---
    ax1 = plt.subplot(2, 2, 1)
    sns.barplot(data=df, x="category", y="latency", hue="category", errorbar="sd", ax=ax1, palette="viridis")
    ax1.set_title("Average Latency (seconds)", fontsize=14)
    ax1.set_xlabel("")
    ax1.set_ylabel("Time (s)")
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f')

    # --- 图 2: 延迟分布 (Box Plot) ---
    ax2 = plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x="category", y="latency", hue="category", ax=ax2, palette="pastel")
    ax2.set_title("Latency Distribution (Stability)", fontsize=14)
    ax2.set_xlabel("")
    ax2.set_ylabel("Time (s)")

    # --- 图 3: 成功率 (Stacked Bar) ---
    ax3 = plt.subplot(2, 2, 3)
    success_counts = df.groupby(['category', 'success']).size().reset_index(name='counts')
    total_counts = df.groupby('category').size().reset_index(name='total')
    success_counts = success_counts.merge(total_counts, on='category')
    success_counts['percentage'] = (success_counts['counts'] / success_counts['total']) * 100
    
    # 只画成功的部分
    success_only = success_counts[success_counts['success']==True]
    if not success_only.empty:
        sns.barplot(data=success_only, x="category", y="percentage", hue="category", ax=ax3, palette="RdYlGn")
        ax3.set_ylim(0, 110)
        for container in ax3.containers:
            ax3.bar_label(container, fmt='%.1f%%')
    ax3.set_title("Intent Recognition Accuracy (%)", fontsize=14)
    ax3.set_xlabel("")
    ax3.set_ylabel("Success Rate (%)")

    # --- 图 4: 思考深度 (Violin Plot) ---
    ax4 = plt.subplot(2, 2, 4)
    sns.violinplot(data=df, x="category", y="thought_len", hue="category", ax=ax4, palette="magma")
    ax4.set_title("Reasoning Depth (Thought Token Length)", fontsize=14)
    ax4.set_xlabel("")
    ax4.set_ylabel("Char Count")

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    img_path = f"{OUTPUT_DIR}/report_{timestamp}.png"
    plt.savefig(img_path, dpi=300)
    print(f"📊 可视化报表已生成: {img_path}")
    
    # 打印简报
    print("\n" + "="*40)
    print("📋 测试摘要 (Summary)")
    print("="*40)
    summary = df.groupby("category")[["latency", "success", "thought_len"]].mean()
    summary.columns = ["Avg Latency (s)", "Success Rate", "Avg Thought Len"]
    print(summary.to_string())

if __name__ == "__main__":
    # 健康检查
    try:
        print("Checking API health...")
        requests.get("http://localhost:8082", timeout=5)
        print("✅ API is online.")
    except:
        print("❌ Error: Cannot connect to http://localhost:8082. Please start app.py first!")
        exit(1)

    # 运行测试
    df_results = run_benchmark()
    
    # 生成报告
    if not df_results.empty:
        generate_report(df_results)
    else:
        print("❌ No data collected.")
