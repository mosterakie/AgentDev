from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Model
# ============================================================

model = ChatDeepSeek(
    model="deepseek-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    }
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


# 用来记录这个 Tool 被调用了几次
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
    # 故意制造错误：
    #
    # 第一次调用直接抛异常
    # 第二次调用正常
    # ========================================================

    # if get_user_detail_call_count == 1:
    raise RuntimeError("数据库连接失败，请稍后重试")

    users = {
        1001: "张三，软件工程师，年龄 25 岁",
        1002: "李四，游戏开发者，年龄 27 岁",
    }

    detail = users.get(user_id)

    if detail is None:
        return "用户不存在"

    return detail


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
# LLM + Tools
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

如果工具执行失败，不要直接认为任务失败。
请根据工具返回的错误信息决定下一步应该怎么做。

如果可以通过重新调用工具完成任务，可以尝试重新调用。

只有在确实无法完成任务时，才向用户说明失败原因。
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

        # 非常重要：
        # 必须保存完整 AIMessage
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

            # --------------------------------------------
            # 查找 Tool
            # --------------------------------------------

            tool = tool_map.get(tool_name)

            if tool is None:

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

            # --------------------------------------------
            # 执行 Tool
            #
            # 这里是本次实验最重要的地方
            # --------------------------------------------

            try:

                result = tool.invoke(tool_args)

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

                # ========================================
                # 不让异常冲出 Agent Loop
                # ========================================

                error_message = (
                    f"工具执行失败：{type(e).__name__}: {e}"
                )

                print("Tool Exception:")
                print(error_message)

                # 把异常转换成 ToolMessage
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
