import os
from typing import Annotated
from dotenv import load_dotenv
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# LangGraph 核心组件
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 引入写好的加载工具（加载本地向量库）
from vectorstore_utils import get_retriever

#联网工具
import dashscope
from dashscope import Generation


load_dotenv()

# ---------- 1. 初始化模型与检索器 ----------
llm = ChatOpenAI(
    model="qwen3.7-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.3,
    max_tokens=1024,
    request_timeout=60,  # 60秒没响应就强行报错，防止无限挂起
)

# 获取检索器
retriever = get_retriever(k=2)

# ---------- 2. 定义工具 (Tools) ----------
@tool
def search_tcm_knowledge(query: str) -> str:
    """当用户询问中医、中药、疾病症状、调理方法等问题时，使用此工具查询本地中医知识库。"""
    
    # ---------- 查询改写（预处理）----------
    rewrite_prompt = f"""请将以下用户查询改写为适合在中医知识库中检索的完整查询语句。
要求：
1. 补全省略的指代（例如"那风热感冒呢？"应改写为"风热感冒的症状和调理方法"）
2. 使用规范的中医术语（例如"怕冷"改为"恶寒"）
3. 保持原意不变
4. 只输出改写后的查询语句，不要任何解释

用户查询：{query}
改写后的查询："""
    
    try:
        rewritten = llm.invoke(rewrite_prompt).content.strip()
        print(f">>> [DEBUG] 查询改写: {query}")
        print(f">>> [DEBUG] 改写后: {rewritten}")
    except Exception as e:
        print(f">>> [DEBUG] 查询改写失败，使用原始查询: {e}")
        rewritten = query
    
    # ---------- 检索 ----------
    docs = retriever.invoke(rewritten)
    if not docs:
        return "本地知识库中未找到相关内容。"
    
    # ---------- 格式化输出（带引用标记）----------
    formatted_docs = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源").split("/")[-1].split("\\")[-1]
        formatted_docs.append(f"[{i}] 来源: {source}\n内容: {doc.page_content}")
    
    return "\n\n".join(formatted_docs)


@tool
def search_web(query: str) -> str:
    """当本地中医知识库中没有相关信息时，使用此工具进行联网搜索。"""
    try:
        # 利用 DashScope 的 API 调用开启联网搜索的模型
        response = Generation.call(
            model="qwen-plus",  # 建议使用 qwen-plus 或 qwen-max，对联网整合效果更好
            messages=[{"role": "user", "content": query}],
            enable_search=True, # 🚨 开启百炼的联网搜索功能
            result_format="message"
        )
        if response.status_code == 200:
            # 返回联网搜索并总结后的结果
            return response.output.choices[0].message.content
        else:
            return f"联网搜索失败：{response.code} - {response.message}"
    except Exception as e:
        return f"联网搜索异常：{str(e)}"

@tool
def calculate_bmi(weight_kg: float, height_cm: float) -> str:
    """计算 BMI 指数。输入体重（公斤）和身高（厘米），返回 BMI 值及健康评估。"""
    if height_cm <= 0 or weight_kg <= 0:
        return "身高和体重必须为正数。"
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    
    if bmi < 18.5:
        status = "偏瘦"
    elif bmi < 24:
        status = "正常范围"
    elif bmi < 28:
        status = "超重"
    else:
        status = "肥胖"
    return f"您的 BMI 为 {bmi:.1f}，属于{status}。建议结合中医体质进行调理。"

    

tools = [search_tcm_knowledge, search_web, calculate_bmi] 
#把工具打包给大模型
llm_with_tools = llm.bind_tools(tools)


# ---------- 3. 定义状态 (State) ----------
class State(TypedDict):
    # 只需要 messages，工具的输出会自动作为 ToolMessage 追加进来
    messages: Annotated[list[BaseMessage], add_messages]


# ---------- 4. 定义节点 (Nodes) ----------
def agent_node(state: State):
    """大模型思考节点：决定是直接回答，还是调用工具"""
    # 构造系统提示词
    system_prompt = """你是一个专业的医疗知识助手，可以回答用户关于疾病、药物、健康生活方式等问题。
请注意：
- 你的回答仅供参考，不能替代专业医生的诊断和治疗建议。
- 仅做医学知识科普，**绝对不能诊断疾病、不开药方，不指导用药**。
- 如果用户描述紧急症状，请立即建议就医或拨打急救电话。
- 保持回答科学、客观、易懂，避免使用过于专业的术语。
- 不要编造不存在的医学结论，不确定就直接诚实说明，并建议咨询专业医生。
- 你拥有两个工具 *search_tcm_knowledge* 查本地中医知识库 和 *search_web* 联网搜索 。

【工作流程】
1. 优先使用 search_tcm_knowledge 查询本地知识库。
2. 如果本地知识库返回“未找到相关内容”，请立刻使用 search_web 进行联网搜索。
3. 如果联网搜索也无法找到答案，请诚实告知用户“根据现有资料无法回答这个问题”，绝不编造。

【引用规则】
- 如果你使用了 search_tcm_knowledge 的内容来回答，必须且仅在对应句子的末尾加上引用标记，例如 [1]。
- 如果使用的是 search_web 的内容，请注明“根据网络资料”。
"""
    # 把系统提示词放到消息列表最前面
    messages_to_send = [SystemMessage(content=system_prompt)] + state["messages"]
    # 调用大模型（带工具）
    response = llm_with_tools.invoke(messages_to_send)
    # 返回新消息
    return {"messages": [response]}


# ---------- 5. 定义条件边 (Conditional Edges) ----------
def should_continue(state: State):
    """判断大模型是否要调用工具"""
    last_message = state["messages"][-1]
    # 如果大模型生成了 tool_calls，说明它要调用工具
    if last_message.tool_calls:
        return "tools"
    # 否则，说明它准备直接回答用户
    return END

# ---------- 6. 构建图 (Graph) ----------
workflow = StateGraph(State)

# 添加节点
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))  # LangGraph 内置的 ToolNode 帮你执行工具

# 设置边
workflow.add_edge(START, "agent")
# 条件边：从 agent 出发，根据 should_continue 的结果决定去向
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
# 工具执行完后，必定回到 agent 继续思考
workflow.add_edge("tools", "agent")

# 编译图，加入记忆持久化
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
