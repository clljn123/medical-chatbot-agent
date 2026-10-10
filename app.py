import os
import uuid
import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# 导入 LangGraph 编译好的 app
from graph_agent import app as agent_app

load_dotenv()

# ---------- 1. 用来给 Gradio 下拉框提供标题的字典 ----------
conversations = {}

# ---------- 2. 从 LangGraph 读取历史并转换为 Gradio 格式 ----------
def format_langgraph_history(session_id):
    """从 LangGraph 的 MemorySaver 中读取历史，并转换为 Gradio 能显示的格式"""
    if not session_id:
        return []
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = agent_app.get_state(config)
        if not state or not state.values or "messages" not in state.values:
            return []
        
        history_list = []
        for msg in state.values["messages"]:
            # 过滤掉系统消息和工具消息
            if isinstance(msg, HumanMessage):
                history_list.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage) and msg.content:
                history_list.append({"role": "assistant", "content": msg.content})
        return history_list
    except Exception as e:
        print(f"读取历史记录失败: {e}")
        return []

# ---------- 3. 会话管理函数 ----------
def get_conversation_choices():
    """返回 Dropdown 需要的 [(显示标题, session_id), ...]"""
    return [(data["title"], sid) for sid, data in conversations.items()]

def new_conversation():
    """新建对话"""
    # 先检查是否已经有空的新会话
    for sid, data in conversations.items():
        if data["title"] == "新对话":
            return sid, [], gr.update(choices=get_conversation_choices(), value=sid)
    
    new_id = str(uuid.uuid4())
    conversations[new_id] = {"title": "新对话"}
    return new_id, [], gr.update(choices=get_conversation_choices(), value=new_id)

def switch_conversation(session_id):
    """切换会话"""
    if not session_id:
        return None, []
    gradio_history = format_langgraph_history(session_id)
    return session_id, gradio_history

def delete_conversation(session_id):
    """删除当前选中的对话"""
    if session_id and session_id in conversations:
        # 1. 从 conversations 字典中删除标题记录
        del conversations[session_id]
        
        # 2. 尝试清理 LangGraph 后台的 MemorySaver 内存
        # 注意：MemorySaver 目前没有直接提供 delete 方法，
        # 但我们可以通过改变 thread_id 的逻辑让它自然被淘汰（详见下方说明）
        
        # 3. 更新 UI 状态
        choices = get_conversation_choices()
        if choices:
            # 如果还有剩余对话，自动切换到第一个
            new_sid = choices[0][1]
            return new_sid, format_langgraph_history(new_sid), gr.update(choices=choices, value=new_sid)
        else:
            # 如果没有对话了，清空状态
            return None, [], gr.update(choices=[], value=None)
    
    # 如果没选任何对话，或者 ID 不存在，保持原样
    return session_id, format_langgraph_history(session_id), gr.update()



# ---------- 4. 聊天核心函数 ----------
def chat(message, chat_history, session_id):
    if not message.strip():
        return "", chat_history, session_id, gr.update()
    
    is_new_session = False
    if not session_id or session_id not in conversations:
        session_id = str(uuid.uuid4())
        conversations[session_id] = {"title": "新对话"}
        is_new_session = True

    # 第一轮对话用问题作为标题
    if conversations[session_id]["title"] == "新对话":
        conversations[session_id]["title"] = message.replace("\n", " ")[:20]
        is_new_session = True

    # ---------- 准备调用 LangGraph ----------
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 10   # 🚨 防止死循环
    }
    
    # 🚨 ：必须在 try 外面定义好 input_message
    input_message = {"messages": [("user", message)]} 

    #输出运行过程，报错的话是出错在哪一步。
    try:
        print(f"\n>>> [DEBUG] 开始调用 Agent，session_id: {session_id}")
        # 使用 invoke 等待完整结果
        final_state = agent_app.invoke(input_message, config)
        print(f">>> [DEBUG] Agent 返回成功!")
        # 最后一条消息就是 AI 的最终回答
        response = final_state["messages"][-1].content
    except Exception as e:
        print(f">>> [DEBUG] Agent 发生异常: {str(e)}")
        response = f"模型调用异常：{str(e)}"


    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response})

    if is_new_session:
        return "", chat_history, session_id, gr.update(choices=get_conversation_choices(), value=session_id)
    else:
        return "", chat_history, session_id, gr.update()


# ---------- 5. Gradio 前端界面 ----------
with gr.Blocks() as demo:
    gr.Markdown("# 医疗知识问答助手 V3 (Agent版)")
    session_state = gr.State(None)

    with gr.Row():
        # 左侧边栏 (比例占 1/4)
        with gr.Column(scale=1):
            conversation_list = gr.Dropdown(
                label="会话列表",
                choices=[],
                interactive=True
            )
            
           #把“新对话”和“删除对话”并排放在同一行
            with gr.Row():
                new_btn = gr.Button("新对话", variant="primary", scale=1)
                delete_btn = gr.Button("删除对话", variant="stop", scale=1)

        # 右侧聊天区 (比例占 3/4)
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="对话", height=500)
            msg = gr.Textbox(label="输入", placeholder="请输入你的健康问题...")

    # ---------- 事件绑定 ----------
    # 新建对话
    new_btn.click(
        new_conversation,
        None,
        [session_state, chatbot, conversation_list]
    )

    # 删除当前对话
    delete_btn.click(
        delete_conversation,
        [session_state],
        [session_state, chatbot, conversation_list]
    )

    # 切换会话
    conversation_list.change(
        switch_conversation,
        [conversation_list],
        [session_state, chatbot]
    )

    # 发送消息
    msg.submit(
        chat,
        [msg, chatbot, session_state],
        [msg, chatbot, session_state, conversation_list]
    )

demo.launch()