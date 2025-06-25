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
        ("system", "现在你是一个智能图书管理AI，可以通过图书名称或者图书简介为借阅读者提供咨询。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
)

# 创建输出解析器
parser = StrOutputParser()

# 构建基础链
chain = prompt | model | parser


# FastAPI
app = FastAPI(
    title="图书管理系统",
    description="支持多轮对话的图书借阅系统的AI助手",
    version="1.0.0"
)

# 网络服务注册
add_routes(
    app,
    chain,
    path="/book_manager",
)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=6007)