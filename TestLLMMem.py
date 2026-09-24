from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, AIMessage
# from os import getenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

from langchain_core.callbacks import BaseCallbackHandler

class DebugHandler(BaseCallbackHandler):

    def on_chat_model_start(
        self,
        serialized,
        messages,
        **kwargs
    ):
        print("\n========== Chat Model Start ==========")

        print("\nSerialized:")
        print(serialized)

        print("\nMessages:")
        print(messages)

        print("\nKwargs:")
        print(kwargs)

        print("======================================\n")


@tool
def is_fibonacci(n: int) -> bool:
    """判断一个整数是否是斐波那契数。"""
    # return True
    if n < 0:
        return False
    a, b = 0, 1
    while a < n:
        a, b = b, a + b
    return a == n

model = ChatDeepSeek(
    model="deepseek-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # api_key = os.getenv("DEEPSEEK_API_KEY"),
)

model_with_tools = model.bind_tools([
    is_fibonacci
])

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

    ai_msg = model_with_tools.invoke(messages,
    config={
        "callbacks": [DebugHandler()]
    })

    
    if ai_msg.tool_calls:
        tool_call = ai_msg.tool_calls[0]
        result = is_fibonacci.invoke(
            tool_call["args"]
        )
        print("Tool Call:", tool_call)
        print("Tool Result:", result)

        history.append(ai_msg)

        history.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )
        messages = [
            ("system", "你是一个中文语音助手，请根据用户的输入返回相应的文本"),
            *history,
        ]

        ai_msg = model_with_tools.invoke(messages)
    

    print("AI：", ai_msg.content)

    # 保存本轮历史
    history.append(("human", user_input))
    history.append(("ai", ai_msg.content))

    