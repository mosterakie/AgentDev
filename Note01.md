# LLM 本身的记忆测试

#### 实验代码

```
from langchain_deepseek
import ChatDeepSeek 
model = ChatDeepSeek( model="deepseek-flash", temperature=0, max_tokens=None, timeout=None, max_retries=2, api_key="sk-", # other params... ) 

# 运行代理 
messages = [ ("system", "你是一个中文语音助手，请根据用户的输入返回相应的文本"), ("human", "你好，还记得我是谁吗"), ] 
ai_msg = model.invoke(messages) 
print(ai_msg+"/n") 
print(ai_msg.content)
```

## 一、实验目的

测试单次 LLM 模型交互是否具有跨请求的记忆能力。

实验过程中，每次调用 `model.invoke(messages)` 时都重新构造 `messages`，不主动向当前请求中加入之前的对话记录。

------

# 二、实验一：测试上一轮输出是否能够被记住

## 第一次运行

### 输入

```python
messages = [
    ("system", "你是一个中文语音助手，请根据用户的输入返回相应的文本"),
    ("human", "你好，能说“最喜欢阮梅了”吗？"),
]
```

### LLM 输出

```text
可以呀，最喜欢阮梅了！
```

### 关键运行信息

```text
model_provider:
    deepseek

model_name:
    deepseek-flash

finish_reason:
    stop

token_usage:
    prompt_tokens: 55
    completion_tokens: 183
    total_tokens: 238

reasoning_tokens:
    174
```

### 说明

本次请求中，LLM 根据当前 `messages` 中的用户输入生成了：

```text
可以呀，最喜欢阮梅了！
```

------

## 第二次运行

### 输入

```python
messages = [
    ("system", "你是一个中文语音助手，请根据用户的输入返回相应的文本"),
    ("human", "你好，请返回上一个输出"),
]
```

### LLM 输出

```text
抱歉，我这边没有找到上一个输出。
当前对话刚开始，还没有可返回的内容。
你可以把想重复的内容再发我一次。
```

### 关键运行信息

```text
model_provider:
    deepseek

model_name:
    deepseek-flash

finish_reason:
    stop

token_usage:
    prompt_tokens: 49
    completion_tokens: 166
    total_tokens: 215

reasoning_tokens:
    137
```

### 实验观察

第二次请求中的 `messages` 只有：

```text
system
human：你好，请返回上一个输出
```

并没有包含第一次请求中的：

```text
human：你好，能说“最喜欢阮梅了”吗？
assistant：可以呀，最喜欢阮梅了！
```

因此模型无法获得第一次交互的信息。

------

# 三、实验二：测试用户身份是否能够被记住

## 第三次运行

### 输入

```python
messages = [
    ("system", "你是一个中文语音助手，请根据用户的输入返回相应的文本"),
    ("human", "你好，我叫爱洗澡的猫"),
]
```

### LLM 输出

```text
你好，爱洗澡的猫！很高兴认识你。
有什么我可以帮你的吗？
```

### 关键运行信息

```text
model_provider:
    deepseek

model_name:
    deepseek-flash

finish_reason:
    stop

token_usage:
    prompt_tokens: 50
    completion_tokens: 101
    total_tokens: 151

reasoning_tokens:
    83
```

### 实验观察

模型能够在**当前请求**中读取：

```text
你好，我叫爱洗澡的猫
```

因此能够在当前回复中使用：

```text
爱洗澡的猫
```

但是，这并不能说明模型已经将该信息保存为长期记忆。

------

## 第四次运行

### 输入

```python
messages = [
    ("system", "你是一个中文语音助手，请根据用户的输入返回相应的文本"),
    ("human", "你好，还记得我是谁吗"),
]
```

### LLM 输出

```text
你好！很抱歉，我这边没有之前的记忆，
所以暂时不知道你是谁。
你可以告诉我你的名字或怎么称呼你吗？
```

### 关键运行信息

```text
model_provider:
    deepseek

model_name:
    deepseek-flash

finish_reason:
    stop

token_usage:
    prompt_tokens: 49
    completion_tokens: 131
    total_tokens: 180

reasoning_tokens:
    104
```

### 实验观察

第三次请求中曾经出现：

```text
你好，我叫爱洗澡的猫
```

但第四次请求重新构造了 `messages`：

```python
messages = [
    ("system", "你是一个中文语音助手，请根据用户的输入返回相应的文本"),
    ("human", "你好，还记得我是谁吗"),
]
```

第四次请求没有包含第三次请求的消息，因此模型无法知道：

```text
爱洗澡的猫
```

是用户之前提供的名字。

------

# 四、实验结果

| 实验   | 测试内容       | 前一次消息是否传入 | LLM 是否能够获得前一次信息 |
| ------ | -------------- | ------------------ | -------------------------- |
| 实验一 | 记住上一次输出 | 否                 | 否                         |
| 实验二 | 记住用户名字   | 否                 | 否                         |

可以观察到：

```text
第一次请求
    ↓
LLM
    ↓
产生输出
    ↓
请求结束
    ↓
第二次请求
    ↓
重新构造 messages
    ↓
LLM
```

第二次请求并没有自动获得第一次请求的：

```text
messages
AIMessage
```

因此，当前测试中没有体现出跨请求记忆。

------

# 五、核心结论

本实验验证的是：

> **单独调用 `model.invoke()` 时，LLM 本身不会因为上一次 API 调用而自动获得完整的对话历史。**

LLM 每次生成回答时，能够使用的信息主要来自**当前请求发送给它的上下文**。

例如第一次：

```python
messages = [
    ("system", "..."),
    ("human", "你好，我叫爱洗澡的猫"),
]
```

LLM 可以知道：

```text
用户叫“爱洗澡的猫”
```

但如果下一次变成：

```python
messages = [
    ("system", "..."),
    ("human", "你好，还记得我是谁吗"),
]
```

那么对于这个请求而言，之前的：

```text
你好，我叫爱洗澡的猫
```

并不存在于输入上下文中。

因此 LLM 无法凭空恢复这个信息。

------

# 六、对 Agent Memory 的启示

这也说明了为什么 Agent 系统通常需要额外的 Memory 机制。

最简单的实现方式就是手动保存历史：

```text
第一次请求

User
    ↓
LLM
    ↓
AIMessage
    ↓
保存


第二次请求

历史消息
    +
新的 User Message
    ↓
LLM
    ↓
新的 AIMessage
```

例如：

```python
messages = [
    ("system", "你是一个中文语音助手"),
    ("human", "你好，我叫爱洗澡的猫"),
    ("ai", "你好，爱洗澡的猫！很高兴认识你。"),
    ("human", "你好，还记得我是谁吗"),
]
```

此时 LLM 才能够根据上下文回答：

```text
你叫爱洗澡的猫。
```

因此可以把 Memory 理解成：

```text
             ┌──────────────┐
             │     User     │
             └──────┬───────┘
                    ↓
             ┌──────────────┐
             │     LLM      │
             └──────┬───────┘
                    ↓
             ┌──────────────┐
             │ AIMessage    │
             └──────┬───────┘
                    ↓
             ┌──────────────┐
             │    Memory    │
             └──────┬───────┘
                    │
                    │ 保存历史
                    ↓
             ┌──────────────┐
             │ 下一次请求    │
             └──────┬───────┘
                    ↓
          历史消息 + 新消息
                    ↓
             ┌──────────────┐
             │     LLM      │
             └──────────────┘
```

所以，**Memory 并不是让 LLM 本身突然拥有了记忆能力，而是由 Agent/应用层负责保存、检索并重新提供上下文。**