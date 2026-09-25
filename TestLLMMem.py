from langchain_deepseek import ChatDeepSeek

from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
)

from langchain_core.tools import tool

from ContextManager import ContextManager


# ================================================================
# 环境变量
# ================================================================

try:
    from dotenv import load_dotenv

    load_dotenv()

except ImportError:
    pass


# ================================================================
# 结构化输出
# ================================================================
from pydantic import BaseModel, Field


# 使用结构输出，让大模型在回答时自己判断是否压缩
class AgentResponse(BaseModel):
    content: str = Field(
        description="给用户的最终回答"
    )

    need_compression: bool = Field(
        description="当前历史上下文是否需要进行语义压缩"
    )


# ================================================================
# Model
# ================================================================

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


# ================================================================
# Tool
# ================================================================

@tool
def is_fibonacci(n: int) -> bool:
    """判断一个整数是否是斐波那契数。"""

    print("Tool Called")
    if n < 0:
        return False

    a, b = 0, 1

    while a < n:
        a, b = b, a + b

    return a == n


# ================================================================
# Tool Model
# ================================================================

model_with_tools = model.bind_tools([
    is_fibonacci
])

structured_model = model_with_tools.with_structured_output(AgentResponse)


# ================================================================
# Summary
# ================================================================

def summarize(
    old_summary: str,
    new_history: str,
) -> str:

    prompt = f"""
你是一个 Agent 的历史压缩器。

你的任务是将：

1. 已有历史摘要
2. 新增的历史对话

整合成一份新的、简洁、准确的长期摘要。

要求：

1. 保留用户明确告诉我们的个人信息
2. 保留重要事实
3. 保留正在进行的任务
4. 保留重要决策
5. 保留重要的工具调用及其结果
6. 删除无意义的闲聊
7. 不要虚构任何信息
8. 不要重复相同的信息
9. 不要描述“你正在进行摘要”
10. 直接输出新的历史摘要

====================
已有历史摘要
====================

{old_summary if old_summary else "无"}

====================
新增历史
====================

{new_history}

====================
新的历史摘要
====================
"""

    response = model.invoke([
        ("human", prompt)
    ])

    return response.content


# ================================================================
# Context Manager
# ================================================================

ctx = ContextManager(
    # 为了方便测试，故意设置得很小
    max_tokens=100,

    # 只保留最近 3 条消息
    keep_recent=3,

    summarizer=summarize,
)


# ================================================================
# System Prompt
# ================================================================

SYSTEM_PROMPT = (
    "你是一个中文语音助手，"
    "请根据用户的输入返回相应的文本。"
)


# ================================================================
# Debug
# ================================================================

def print_context():

    print("\n========== Context ==========")

    print("Summary:")

    if ctx.summary:
        print(ctx.summary)
    else:
        print("(空)")

    print("\nRecent History:")

    for i, message in enumerate(ctx.history):

        print(
            f"[{i}] "
            f"{type(message).__name__}: "
            f"{message.content}"
        )

    print(
        "\nEstimated Tokens:",
        ctx._count_tokens(ctx.history)
    )

    print(
        "Should Compress:",
        ctx.should_compress()
    )

    print("==============================\n")




# ================================================================
# Main
# ================================================================

while True:

    user_input = input("你：")

    # ------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------

    if user_input.lower() in [
        "exit",
        "quit",
        "q",
    ]:

        break

    # ------------------------------------------------------------
    # 查看 Context
    # ------------------------------------------------------------

    if user_input.lower() == "history":

        print_context()

        print("发送给 LLM 的完整 Context：")

        for message in ctx.get_messages():

            print(
                f"{type(message).__name__}: "
                f"{message.content}"
            )

        print()

        continue

    # ------------------------------------------------------------
    # 1. 用户消息进入 Context
    # ------------------------------------------------------------

    ctx.add_message(
        HumanMessage(
            content=user_input
        )
    )

    # ------------------------------------------------------------
    # 2. 第一次调用 LLM
    # ------------------------------------------------------------

    messages = [
        ("system", SYSTEM_PROMPT),
        *ctx.get_messages(),
    ]

    ai_msg = model_with_tools.invoke(
        messages
    )

    # ------------------------------------------------------------
    # 3. 保存 AIMessage
    #
    # 注意：
    # 这里保存完整 AIMessage
    #
    # 不能只保存 ai_msg.content
    #
    # 因为 tool_calls 也属于上下文。
    # ------------------------------------------------------------

    ctx.add_message(ai_msg)

    # ------------------------------------------------------------
    # 4. Tool Calling
    # ------------------------------------------------------------

    if ai_msg.tool_calls:

        for tool_call in ai_msg.tool_calls:

            print(
                "Tool Call:",
                tool_call
            )

            # --------------------------------------------
            # 根据工具名称选择工具
            # --------------------------------------------

            if tool_call["name"] == "is_fibonacci":

                result = is_fibonacci.invoke(
                    tool_call["args"]
                )

            else:

                result = (
                    f"未知工具："
                    f"{tool_call['name']}"
                )

            # --------------------------------------------
            # Tool Result
            # --------------------------------------------

            print(
                "Tool Result:",
                result
            )

            # --------------------------------------------
            # 保存 ToolMessage
            # --------------------------------------------

            ctx.add_message(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )

        # --------------------------------------------------------
        # 5. Tool 执行结束后再次调用 LLM
        # --------------------------------------------------------

        messages = [
            ("system", SYSTEM_PROMPT),
            *ctx.get_messages(),
        ]

        ai_msg = structured_model.invoke(
            messages
        )

        # --------------------------------------------------------
        # 保存最终 AIMessage
        # --------------------------------------------------------

        ctx.add_message(ai_msg)

    # ------------------------------------------------------------
    # 6. 输出最终答案
    # ------------------------------------------------------------

    print(
        "AI：",
        ai_msg
    )

    # ------------------------------------------------------------
    # 7. 当前 Agent Step 完成
    #
    # 现在才检查是否需要压缩
    #
    # 注意：
    # 不放在 get_messages() 里面。
    # ------------------------------------------------------------

    if ctx.should_compress():

        print(
            "\n[ContextManager] "
            "History 超过阈值，开始压缩..."
        )

        ctx.compress()

        print(
            "[ContextManager] "
            "压缩完成。"
        )

