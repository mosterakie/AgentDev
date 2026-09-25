# 第五次实验 - Minimal Agent Runtime

# 一、阶段目标

本阶段的目标不是直接使用 LangChain 的高级 Agent API，而是通过手写最小 Agent Runtime，理解 Agent 最基础的运行机制。

核心问题：

> 一个 Agent 到底是由什么组成的？

经过本阶段实验，得到的初步模型是：

```
Agent
│
├── Model
├── Tools
├── State
└── Runtime
```

# 二、LLM 本身没有记忆

最初通过两组独立 LLM 请求验证：

```
Request 1
    ↓
LLM
    ↓
Response 1

Request 2
    ↓
LLM
    ↓
Response 2
```

如果第二次请求没有携带第一次的信息，LLM 不知道第一次发生了什么。

因此：

```
LLM ≠ Memory
```

所谓 Agent Memory，本质上需要由应用程序保存状态，并在后续请求中重新提供给 LLM。

# 三、Messages 是 LLM 上下文的基础

Agent 运行过程中不能只保存：

```
ai_msg.content
```

而应该保存完整的消息：

```
HumanMessage
AIMessage
ToolMessage
```

尤其是：

```
AIMessage.tool_calls
```

它包含了 LLM 请求 Runtime 执行工具的动作信息。

因此 Agent State 当前最核心的数据就是：

```
state.messages
```

# 四、Tool Calling

通过：

```
model.bind_tools(tools)
```

可以将工具定义提供给 LLM。

但是：

```
bind_tools()
```

本身不会执行 Python 函数。

真正的流程是：

```
User
 ↓
LLM
 ↓
AIMessage
 ↓
tool_calls
 ↓
Runtime
 ↓
Tool
 ↓
ToolMessage
 ↓
LLM
```

因此：

> Tool Calling ≠ Tool Execution

LLM 负责决定调用什么工具，Runtime 负责实际执行。

# 五、最小 Agent Loop

最核心的 Agent Loop 可以简化为：

```
while True:

    response = LLM(messages)

    messages.append(response)

    if not response.tool_calls:
        break

    for tool_call in response.tool_calls:

        result = execute_tool(tool_call)

        messages.append(result)
```

这几行代码实际上已经构成了最基本的 Agent。

流程：

```
              LLM
               │
               ▼
          AIMessage
               │
        ┌──────┴──────┐
        │             │
     Tool Call      No Tool
        │             │
        ▼             ▼
      Tool           END
        │
        ▼
   ToolMessage
        │
        └────────→ LLM
```

# 六、Multiple Tool Calls

一个 LLM Response 可以同时产生多个 Tool Call。

例如：

```
用户：
3350 是斐波那契数吗？
同时判断它是不是质数。
```

LLM 可以产生：

```
AIMessage
├── is_fibonacci(3350)
└── is_prime(3350)
```

Runtime 应该执行当前轮产生的所有 Tool Call：

```
for tool_call in ai_msg.tool_calls:
    execute_tool(tool_call)
```

因此：

> 多个 Tool Call 不等于多个 LLM Turn。

一个 LLM Turn 可以产生多个独立 Action。

# 七、Tool Dependency

实验：

```
用户：
张三的信息是什么？
```

实际过程：

```
LLM
 ↓
find_user("张三")
 ↓
1001
 ↓
LLM
 ↓
get_user_detail(1001)
 ↓
用户详细信息
 ↓
LLM
 ↓
Final
```

这里：

```
get_user_detail()
```

的参数：

```
1001
```

是在前一个 Tool 执行后才产生的。

因此：

```
Tool A
 ↓
Result A
 ↓
LLM
 ↓
Tool B
```

这是一个真正的 Tool Dependency。

关键结论：

> Runtime 不需要提前知道 Tool 之间的依赖关系。

Runtime 只需要：

```
执行 LLM 请求的 Action
 ↓
把结果交给 LLM
 ↓
由 LLM 决定下一步
```

因此 Tool Dependency 是由：

```
LLM + State + Tool Result
```

动态形成的。

# 八、Tool Error Handling

Tool 可能执行失败：

```
try:
    result = tool.invoke(args)

except Exception as e:
    ...
```

Runtime 不应该让普通 Tool Exception 直接摧毁整个 Agent Loop。

可以转换成：

```
ToolMessage
    ↓
工具执行失败：RuntimeError: 数据库连接失败
```

然后交给 LLM。

于是：

```
Tool
 ↓
Exception
 ↓
Runtime 捕获
 ↓
ToolMessage(Error)
 ↓
LLM
```

LLM 可以根据错误决定：

```
Retry
```

或者：

```
Stop
```

或者：

```
使用其他 Tool
```

# 九、Retry 的职责

实验发现：

DeepSeek 和 Qwen 在相同 Runtime、相同 Tool 错误条件下表现不同。

DeepSeek 实验中：

```
Tool Error
 ↓
LLM
 ↓
Retry
 ↓
Tool Error
 ↓
LLM
 ↓
Stop
```

Qwen 实验中：

```
Tool Error
 ↓
LLM
 ↓
Stop
```

因此 Retry 并不是 Runtime 自动决定的。

当前设计中：

```
LLM
 ↓
决定是否 Retry
```

而 Runtime：

```
执行 Tool
 ↓
捕获 Error
 ↓
返回 ToolMessage
```

这说明：

> Agent 的行为策略很大程度上来自 LLM。

不过 Runtime 仍然应该存在硬性的安全边界，例如：

```
max_iterations = 10
```

它不是为了让 Agent Retry，而是防止 Agent 无限运行。

# 十、State

原来的实现中：

```
def run(user_input):

    messages = []
```

意味着每次运行都会创建新的消息列表。

后来改成：

```
class AgentState:

    def __init__(self):
        self.messages = []
```

然后：

```
state = runtime.create_state()

runtime.run(
    state,
    user_input
)
```

这样多个 Run 可以共享同一个 State：

```
State
│
└── messages
    │
    ├── 第一次对话
    │
    ├── 第二次对话
    │
    └── 第三次对话
```

因此：

> State 是 Agent 跨 Run 保存状态的载体。

# 十一、State 与 Memory

这里需要区分：

```
State
```

和：

```
Memory
```

State 是更大的概念。

例如：

```
AgentState
├── messages
├── current_task
├── iteration
├── user_id
└── ...
```

而 Memory / Context Management 主要处理：

```
历史信息
```

因此：

```
State
└── ContextManager
```

是目前确定的架构方向。

# 十二、ContextManager

之前已经完成 ContextManager 实验。

它负责：

```
历史消息
 ↓
Token 统计
 ↓
上下文判断
 ↓
语义压缩
 ↓
Summary + Recent Messages
```

因此未来结构可以是：

```
AgentState
│
├── ContextManager
│   │
│   ├── Messages
│   ├── Summary
│   ├── Token Count
│   └── Compression
│
├── Current Task
├── Iteration
└── ...
```

核心关系：

```
State
 ↓
拥有 Agent 的状态

ContextManager
 ↓
管理其中的上下文状态

LLM
 ↓
只接收当前需要的 Context
```

# 十三、当前 Agent Runtime 架构

经过本阶段实验，目前可以抽象为：

```
                         Agent
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
           Model         Tools        Runtime
                                         │
                                         ▼
                                       State
                                         │
                                         ▼
                                  ContextManager
                                         │
                                         ▼
                                      Messages
                                         │
                                         ▼
                                        LLM
```

Runtime 的主要职责：

```
1. 调用 Model
2. 保存 / 读取 State
3. 解析 Tool Call
4. 查找 Tool
5. 执行 Tool
6. 捕获 Tool Exception
7. 创建 ToolMessage
8. 控制 Agent Loop
9. 提供最大迭代次数等安全边界
```

LLM 的主要职责：

```
1. 判断是否需要 Tool
2. 选择 Tool
3. 生成 Tool 参数
4. 理解 Tool Result
5. 判断下一步
6. 判断是否 Retry
7. 最终生成回答
```

Tool 的职责：

```
提供实际能力

例如：

数据库
HTTP API
文件系统
搜索
游戏世界
业务系统
```

# 十四、目前的核心认识

目前可以把 Agent 理解为：

```
Agent
=
Model
+
Tools
+
State
+
Runtime Loop
```

进一步：

```
Model
    决策

Tools
    能力

State
    状态

Runtime
    执行与协调
```

其中最重要的循环是：

```
        ┌──────────────┐
        │     State    │
        └──────┬───────┘
               │
               ▼
             Model
               │
               ▼
           Decision
               │
       ┌───────┴───────┐
       │               │
    Tool Call        Final
       │               │
       ▼               ▼
   Tool Executor      END
       │
       ▼
   Tool Result
       │
       ▼
     State
       │
       └────────────→ Model
```