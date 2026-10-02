from dotenv import load_dotenv

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool


# ============================================================
# 1. Environment
# ============================================================

load_dotenv()


# ============================================================
# 2. Agent State
# ============================================================

class AgentState:
    """
    Agent 的运行状态。

    当前只保存 messages。

    后续可以逐渐加入：
        - iteration
        - context
        - summary
        - user_id
        - current_task
        - ...
    """

    def __init__(self):
        self.messages = []


# ============================================================
# 3. Tools
# ============================================================

@tool
def find_user(name: str) -> int:
    """
    根据用户姓名查找用户 ID。
    找不到时返回 -1。
    """

    users = {
        "张三": 1001,
        "李四": 1002,
    }

    return users.get(name, -1)


@tool
def get_user_detail(user_id: int) -> str:
    """
    根据用户 ID 获取用户详细信息。
    """

    users = {
        1001: "张三，软件工程师，年龄 25 岁",
        1002: "李四，产品经理，年龄 28 岁",
    }

    if user_id not in users:
        raise RuntimeError(f"找不到用户 ID：{user_id}")

    return users[user_id]


@tool
def is_fibonacci(n: int) -> bool:
    """
    判断一个整数是否为斐波那契数。
    """

    a, b = 0, 1

    while a <= n:

        if a == n:
            return True

        a, b = b, a + b

    return False


# ============================================================
# 4. Agent Runtime
# ============================================================

class AgentRuntime:

    def __init__(
        self,
        model,
        tools,
        system_prompt: str,
        max_iterations: int = 10,
    ):
        self.model = model

        self.tools = tools

        self.tool_map = {
            tool.name: tool
            for tool in tools
        }

        self.model_with_tools = model.bind_tools(tools)

        self.system_prompt = system_prompt

        self.max_iterations = max_iterations

    # --------------------------------------------------------
    # 初始化 State
    # --------------------------------------------------------

    def create_state(self) -> AgentState:
        """
        创建一个新的 Agent State。

        每一个独立的 Agent 会话可以拥有自己的 State。
        """

        state = AgentState()

        state.messages = [
            (
                "system",
                self.system_prompt,
            )
        ]

        return state

    # --------------------------------------------------------
    # Tool Executor
    # --------------------------------------------------------

    def execute_tool(self, tool_call) -> ToolMessage:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        print("\n========== TOOL CALL ==========")
        print(f"Tool Name: {tool_name}")
        print(f"Tool Args: {tool_args}")

        # ----------------------------------------------------
        # Tool 不存在
        # ----------------------------------------------------

        if tool_name not in self.tool_map:

            error_message = (
                f"工具执行失败：不存在名为 {tool_name} 的工具"
            )

            print(error_message)

            return ToolMessage(
                content=error_message,
                tool_call_id=tool_call_id,
                name=tool_name,
            )

        tool = self.tool_map[tool_name]

        # ----------------------------------------------------
        # 执行 Tool
        # ----------------------------------------------------

        try:

            result = tool.invoke(tool_args)

            print("\n========== TOOL RESULT ==========")
            print(result)

            return ToolMessage(
                content=str(result),
                tool_call_id=tool_call_id,
                name=tool_name,
            )

        # ----------------------------------------------------
        # 捕获 Tool Exception
        # ----------------------------------------------------

        except Exception as e:

            error_message = (
                f"工具执行失败：{type(e).__name__}: {e}"
            )

            print("\n========== TOOL ERROR ==========")
            print(error_message)

            return ToolMessage(
                content=error_message,
                tool_call_id=tool_call_id,
                name=tool_name,
            )

    # --------------------------------------------------------
    # Agent Loop
    # --------------------------------------------------------

    def run(
        self,
        state: AgentState,
        user_input: str,
    ):

        # ----------------------------------------------------
        # 把用户输入加入 State
        # ----------------------------------------------------

        state.messages.append(
            HumanMessage(
                content=user_input
            )
        )

        # ----------------------------------------------------
        # Agent Loop
        # ----------------------------------------------------

        for iteration in range(
            1,
            self.max_iterations + 1,
        ):

            print("\n")
            print("=" * 60)
            print(
                f"                    ITERATION {iteration}"
            )
            print("=" * 60)

            # ------------------------------------------------
            # LLM
            # ------------------------------------------------

            ai_msg: AIMessage = (
                self.model_with_tools.invoke(
                    state.messages
                )
            )

            print("\n========== LLM ==========")

            print("Content:")
            print(ai_msg.content)

            print("\nTool Calls:")

            if ai_msg.tool_calls:

                for tool_call in ai_msg.tool_calls:
                    print(tool_call)

            else:
                print("None")

            # ------------------------------------------------
            # 保存 AIMessage
            # ------------------------------------------------

            state.messages.append(ai_msg)

            # ------------------------------------------------
            # 没有 Tool Call
            #
            # Agent 认为任务已经完成
            # ------------------------------------------------

            if not ai_msg.tool_calls:

                print("\n========== FINAL ==========")
                print(ai_msg.content)

                return ai_msg.content

            # ------------------------------------------------
            # 执行当前轮所有 Tool Call
            # ------------------------------------------------

            for tool_call in ai_msg.tool_calls:

                tool_message = self.execute_tool(
                    tool_call
                )

                state.messages.append(
                    tool_message
                )

        # ----------------------------------------------------
        # Runtime 安全边界
        # ----------------------------------------------------

        print("\n========== RUNTIME STOP ==========")

        print(
            f"达到最大迭代次数 {self.max_iterations}，"
            "Runtime 强制停止 Agent。"
        )

        return (
            "Agent 达到最大迭代次数，"
            "Runtime 强制停止。"
        )


# ============================================================
# 5. Model
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
# 6. Runtime
# ============================================================

runtime = AgentRuntime(

    model=model,

    tools=[
        find_user,
        get_user_detail,
        is_fibonacci,
    ],

    system_prompt="""
你是一个中文助手，可以使用工具完成任务。

你可以：
1. 根据姓名查询用户 ID
2. 根据用户 ID 查询用户详细信息
3. 判断数字是否为斐波那契数

请遵循以下规则：

- 如果需要工具，就调用工具。
- 不要编造工具返回的数据。
- 如果一个工具的结果是另一个工具所需要的输入，
  请等待前一个工具返回结果后再继续。
- 如果工具执行失败，可以根据错误信息决定是否重试。
- 如果已经获得足够的信息，就直接回答用户。
""",

    max_iterations=10,
)


# ============================================================
# 7. CLI
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("          Minimal Agent Runtime - State")
    print("=" * 60)

    # --------------------------------------------------------
    # 创建一个 Agent State
    # --------------------------------------------------------

    state = runtime.create_state()

    while True:

        user_input = input("\n你：").strip()

        if not user_input:
            continue

        if user_input.lower() in {
            "exit",
            "quit",
            "q",
        }:
            break

        runtime.run(
            state,
            user_input,
        )
