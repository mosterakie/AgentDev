from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv
import os

load_dotenv()


# ============================================================
# Model
# ============================================================

model = ChatOpenAI(
    model="qwen-plus",

    # 阿里云百炼 OpenAI 兼容接口
    base_url="https://ws-qj9smdseynjkcq5d.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",

    # API Key 从 DASHSCOPE_API_KEY 环境变量读取
    api_key=os.getenv("DASHSCOPE_API_KEY"),

    temperature=0,

    max_tokens=None,

    timeout=None,

    max_retries=2,
)


# ============================================================
# Tools
# ============================================================

@tool
def find_user(name: str) -> int:
    """根据用户名查询用户ID。"""

    users = {
        "张三": 1001,
        "李四": 1002,
    }

    user_id = users.get(name)

    if user_id is None:
        return -1

    return user_id


# ============================================================
# 故意制造永久失败
# ============================================================

get_user_detail_call_count = 0


@tool
def get_user_detail(user_id: int) -> str:
    """根据用户ID查询用户详细信息。"""

    global get_user_detail_call_count

    get_user_detail_call_count += 1

    print(
        f"\n[Tool Internal] "
        f"get_user_detail 第 {get_user_detail_call_count} 次调用"
    )

    # ========================================================
    # 永远失败
    # ========================================================

    raise RuntimeError(
        "数据库连接失败，请稍后重试"
    )


# ============================================================
# Tool Map
# ============================================================

tools = [
    find_user,
    get_user_detail,
]

tool_map = {
    tool.name: tool
    for tool in tools
}


# ============================================================
# Bind Tools
# ============================================================

model_with_tools = model.bind_tools(tools)


# ============================================================
# Agent Loop
# ============================================================

def run_agent(user_input: str):

    messages = [
        (
            "system",
            """
你是一个中文助手，可以使用工具查询用户信息。

如果工具执行失败，请根据工具返回的错误信息决定下一步。

如果认为重试有意义，可以重试工具。

如果认为无法继续完成任务，请停止调用工具并向用户说明原因。

不要编造工具返回的数据。
"""
        ),
        (
            "human",
            user_input
        )
    ]

    while True:

        # ====================================================
        # LLM
        # ====================================================

        ai_msg = model_with_tools.invoke(messages)

        print("\n========== LLM ==========")

        print("Content:")
        print(ai_msg.content)

        print("\nTool Calls:")

        for tool_call in ai_msg.tool_calls:
            print(tool_call)

        print("==========================")

        # 保存完整 AIMessage
        messages.append(ai_msg)

        # ====================================================
        # 没有 Tool Call
        # ====================================================

        if not ai_msg.tool_calls:

            print("\nLLM 没有调用工具，结束 Loop。")

            return ai_msg

        # ====================================================
        # 执行 Tool Calls
        # ====================================================

        for tool_call in ai_msg.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            print("\n========== Tool ==========")

            print("Tool Name:")
            print(tool_name)

            print("Tool Args:")
            print(tool_args)

            print("Tool Call ID:")
            print(tool_call_id)

            # ------------------------------------------------
            # 查找 Tool
            # ------------------------------------------------

            selected_tool = tool_map.get(tool_name)

            if selected_tool is None:

                error_message = (
                    f"工具 {tool_name} 不存在"
                )

                print("Tool Error:")
                print(error_message)

                messages.append(
                    ToolMessage(
                        content=error_message,
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                )

                continue

            # ------------------------------------------------
            # 执行 Tool
            # ------------------------------------------------

            try:

                result = selected_tool.invoke(tool_args)

                print("Tool Result:")
                print(result)

                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                )

            except Exception as e:

                # =================================================
                # 关键：
                #
                # 不让 Tool Exception 直接终止 Agent。
                #
                # 把异常转换成 ToolMessage，
                # 然后交还给 LLM。
                # =================================================

                error_message = (
                    f"工具执行失败："
                    f"{type(e).__name__}: {e}"
                )

                print("Tool Exception:")
                print(error_message)

                messages.append(
                    ToolMessage(
                        content=error_message,
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                )

            print("==========================")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    user_input = input("你：")

    result = run_agent(user_input)

    print("\n========== Final Answer ==========")
    print(result.content)
    print("==================================")
