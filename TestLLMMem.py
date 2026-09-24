from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, AIMessage
# from os import getenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


model = ChatDeepSeek(
    model="deepseek-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # api_key = os.getenv("DEEPSEEK_API_KEY"),
)

history = []

while True:
    user_input = input("你：")

    # 退出
    if user_input.lower() in ["exit", "quit", "q"]:
        break

    messages = [
        ("system", "你是一个中文语音助手，请根据用户的输入返回相应的文本"),
        *history,
        ("human", user_input),
    ]

    ai_msg = model.invoke(messages)

    print("AI：", ai_msg.content)

    # 保存本轮历史
    history.append(("human", user_input))
    history.append(("ai", ai_msg.content))

    