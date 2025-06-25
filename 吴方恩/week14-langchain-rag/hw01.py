import os
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langserve import add_routes
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# 全局存储不同会话的历史记录
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """根据session_id获取或创建聊天历史记录"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 加载环境变量
load_dotenv(find_dotenv())

# 创建语言模型
model = ChatOpenAI(
    model='glm-4-flash-250414',
    base_url=os.environ['base_url'],
    api_key=os.environ['api_key'],
    temperature=0.7
)

# 创建提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个拥有丰富编程经验的AI领域的专家，尽可能的回答我的一些AI方面的疑问。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
)

# 创建输出解析器
parser = StrOutputParser()

# 构建基础链
chain = prompt | model | parser

# 创建带历史记录的链
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# 创建FastAPI应用
app = FastAPI(
    title="多轮对话AI助手API",
    description="支持多轮对话和记忆功能的AI助手服务",
    version="1.0.0"
)

# 定义请求模型
class ChatRequest(BaseModel):
    input: str
    session_id: str

# 定义响应模型
class ChatResponse(BaseModel):
    response: str

# 添加自定义路由处理多轮对话
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """处理多轮对话请求"""
    response = chain_with_history.invoke(
        {"input": request.input},
        config={"configurable": {"session_id": request.session_id}}
    )
    return ChatResponse(response=response)

# 添加路由
add_routes(
    app,
    chain_with_history,
    path="/chat_chain",
    input_type=ChatRequest,
    output_type=ChatResponse,
)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=6006)