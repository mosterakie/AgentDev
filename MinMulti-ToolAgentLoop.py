from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# ============================================================
# 1. 环境
# ============================================================

load_dotenv()


# ============================================================
# 2. LLM
# ============================================================

model = ChatDeepSeek(
    model="deepseek-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,

    # 当前实验关闭 Thinking。
    # 原因：
    # DeepSeek Thinking Mode 与 LangChain 当前
    # with_structured_output() 存在 tool_choice 兼容问题。
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    }
)


# ============================================================
# 3. Tools
# ============================================================

@tool
def is_fibonacci(n: int) -> bool:
    """判断一个整数是否是斐波那契数。"""

    if n < 0:
        return False

    a, b = 0, 1

    while a < n:
        a, b = b, a + b

    return a == n


@tool
def is_prime(n: int) -> bool:
    """判断一个整数是否是质数。"""

    if n < 2:
        return False

    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False

    return True

@tool
def factorize(n: int) -> str:
    """对一个正整数进行质因数分解，并返回分解结果。"""

    if n < 2:
        return str(n)

    original = n
    factors = []

    divisor = 2

    while divisor * divisor <= n:

        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor

        divisor += 1

    if n > 1:
        factors.append(n)

    return f"{original} = " + " × ".join(map(str, factors))

@tool
def find_user(name: str) -> int:
    """根据用户名查找用户ID。不存在返回-1"""
    users = {
        "张三": 1001,
        "李四": 1002,
        "王五": 1003,
    }

    return users.get(name, -1)

@tool
def get_user_detail(user_id: int) -> str:
    """根据用户ID查询用户详细信息。"""
    users = {
        1001: "张三，25岁，软件工程师",
        1002: "李四，30岁，产品经理",
        1003: "王五，28岁，设计师",
    }

    return users.get(user_id, "用户不存在")


# ============================================================
# 4. Tool Registry
# ============================================================

tools = [
    is_fibonacci,
    is_prime,
    factorize,
    get_user_detail,
    find_user
]

tool_map = {
    "is_fibonacci": is_fibonacci,
    "is_prime": is_prime,
    "factorize":factorize,
    "get_user_detail":find_user,
    "find_user":find_user,
}


# ============================================================
# 5. Bind Tools
# ============================================================

model_with_tools = model.bind_tools(tools)


# ============================================================
# 6. Structured Output
# ============================================================

class AgentResponse(BaseModel):
    answer: str = Field(
        description="给用户的最终回答"
    )

    need_compression: bool = Field(
        description="当前历史上下文是否需要进行语义压缩"
    )


# ============================================================
# 7. Agent Loop
# ============================================================

def run_agent(user_input: str):

    messages = [
        (
            "system",
            "你是一个中文助手，可以使用工具解决数学问题。"
        ),
        (
            "human",
            user_input
        ),
    ]

    while True:

        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------

        ai_msg = model_with_tools.invoke(messages)

        print("\n========== LLM ==========")
        print("Content:")
        print(ai_msg.content)

        print("\nTool Calls:")
        for tool_call in ai_msg.tool_calls:
            print(tool_call)

        print("==========================")


        # ----------------------------------------------------
        # 保存 AIMessage
        # ----------------------------------------------------

        messages.append(ai_msg)


        # ----------------------------------------------------
        # 没有 Tool Call
        # 说明 LLM 已经可以直接回答
        # ----------------------------------------------------

        if not ai_msg.tool_calls:

            print("\nLLM 没有调用工具，结束 Loop。")

            return ai_msg


        # ----------------------------------------------------
        # 执行所有 Tool Call
        # ----------------------------------------------------

        for tool_call in ai_msg.tool_calls:

            print("\n========== Tool ==========")

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            print("Tool Name:", tool_name)
            print("Tool Args:", tool_args)
            print("Tool Call ID:", tool_call_id)


            # 根据名字找到真正的 Python Tool
            tool = tool_map[tool_name]


            # 执行 Tool
            result = tool.invoke(tool_args)

            print("Tool Result:", result)

            print("==========================")


            # ------------------------------------------------
            # 把 Tool Result 包装成 ToolMessage
            # ------------------------------------------------

            tool_message = ToolMessage(
                content=str(result),
                tool_call_id=tool_call_id,
                name=tool_name,
            )

            messages.append(tool_message)


# ============================================================
# 8. Main
# ============================================================

if __name__ == "__main__":

    user_input = input("你：")

    if user_input.lower() == "q":
        exit()

    response = run_agent(user_input)

    print("\n========== Final Answer ==========")
    print(response.content)
    print("==================================")