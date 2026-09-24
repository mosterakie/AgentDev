## 第三次实验：Tool Calling 与 LLM 二次推理



```

========== Chat Model Start ========== 
Serialized: 
{
	'lc': 1,
	'type': 'not_implemented',
	'id': ['langchain_deepseek', 'chat_models', 'ChatDeepSeek'],
	'repr': "ChatDeepSeek(metadata={'lc_versions': {'langchain-core': '1.6.4', 'langchain': '1.4.2', 'langchain-openai': '1.6.4', 'langchain-deepseek': '1.1.1'}}, profile={'name': 'DeepSeek V4.1 Flash', 'release_date': '2026-09-10', 'last_updated': '2026-09-10', 'open_weights': True, 'max_input_tokens': 1000000, 'max_output_tokens': 384000, 'text_inputs': True, 'image_inputs': True, 'audio_inputs': False, 'video_inputs': False, 'text_outputs': True, 'image_outputs': False, 'audio_outputs': False, 'video_outputs': False, 'reasoning_output': True, 'tool_calling': True, 'structured_output': True, 'attachment': True, 'temperature': True, 'tool_call_streaming': True}, client=<openai.resources.chat.completions.completions.Completions object at 0x000001ACD5A4BAA0>, async_client=<openai.resources.chat.completions.completions.AsyncCompletions object at 0x000001ACD5D80C20>, root_client=<openai.OpenAI object at 0x000001ACD42A0770>, root_async_client=<openai.AsyncOpenAI object at 0x000001ACD55160F0>, model_name='deepseek-flash', temperature=0.0, model_kwargs={}, max_retries=2, stream_chunk_timeout=120.0, api_key=SecretStr('**********'), api_base='https://api.deepseek.com/v1')",
	'name': 'ChatDeepSeek'
}
Messages: [
	[SystemMessage(content = '你是一个中文语音助手，请根据用户的输入返回相应的文本', 		
	additional_kwargs = {},
    response_metadata = {}), 
    HumanMessage(content = '133是斐波拉契数吗', 
    additional_kwargs = {}, 
    response_metadata = {})]
] Kwargs: {
	'run_id': UUID('01a0d297-bfec-7221-b641-849f313ff7f2'),
	'parent_run_id': None,
	'tags': [],
	'metadata': {
		'ls_provider': 'deepseek',
		'ls_model_name': 'deepseek-flash',
		'ls_model_type': 'chat',
		'ls_temperature': 0.0,
		'ls_integration': 'langchain_chat_model',
		'lc_versions': {
			'langchain-core': '1.6.4',
			'langchain': '1.4.2',
			'langchain-openai': '1.6.4',
			'langchain-deepseek': '1.1.1'
		}
	},
	'invocation_params': {
		'model': 'deepseek-flash',
		'model_name': 'deepseek-flash',
		'stream': False,
		'temperature': 0.0,
		'_type': 'chat-deepseek',
		'stop': None,
		'tools': [{
			'type': 'function',
			'function': {
				'name': 'is_fibonacci',
				'description': '判断一个整数是否是斐波那契数。',
				'parameters': {
					'properties': {
						'n': {
							'type': 'integer'
						}
					},
					'required': ['n'],
					'type': 'object'
				}
			}
		}]
	},
	'options': {
		'stop': None
	},
	'name': None,
	'batch_size': 1
}
======================================
```



### 1. 实验目的

在前两次实验的基础上，引入一个简单的 Fibonacci 判断工具：

```
@tool
def is_fibonacci(n: int) -> bool:
    """判断一个整数是否是斐波那契数。"""
    ...
```

观察完整的 Tool Calling 流程：

```
用户输入
  ↓
LLM
  ↓
tool_calls
  ↓
Python 执行 Tool
  ↓
Tool Result
  ↓
LLM
  ↓
最终回答
```

重点观察：

> **LLM 得到 Tool Result 后，会不会只使用 Tool 返回的信息，还是会继续自行推理。**

------

### 2. 实验过程

用户输入：

```
233是一个斐波那契数吗
```

LLM 产生：

```
{
    "name": "is_fibonacci",
    "args": {"n": 233},
    ...
}
```

程序执行：

```
is_fibonacci(233)
```

Tool 只返回：

```
True
```

但最终 LLM 回答：

```
233 是斐波那契数。

它正好是斐波那契数列中的第 13 项
（按 F₁=1, F₂=1 计算）。
```

其中 **“第 13 项”并没有来自 Tool**。

------

### 3. 进一步实验

为了排除 Tool 自身计算的问题，将 `is_fibonacci` 临时修改为始终返回：

```
True
```

输入：

```
156是一个斐波那契数吗
```

Tool：

```
True
```

LLM：

```
156 是斐波那契数。
```

这证明最终回答中的事实判断确实受到了 Tool Result 的影响。

随后又将 Tool 修改为始终返回：

```
False
```

再次输入：

```
156是一个斐波那契数吗
```

Tool：

```
False
```

LLM：

```
156 不是斐波那契数。

斐波那契数列中接近它的数是 144 和 233。
```

这里的：

```
144
233
```

同样不是 Tool 返回的，而是 LLM 根据上下文自行生成的额外信息。

------

## 4. 实验结论

这次实验验证了一个非常重要的 Agent 特性：

> **Tool Result 是 LLM 后续推理的输入，而不是对 LLM 输出的硬约束。**

也就是说：

```
Tool：
True
```

并不意味着：

```
LLM：
只能输出 True
```

而是：

```
Tool：
True
   ↓
LLM：
根据 True + 用户问题 + 自身知识
生成最终回答
```

因此，LLM 仍然可能：

- 解释 Tool Result
- 补充额外信息
- 进行进一步推理
- 甚至在 Tool Result 不完整时自行补全信息

------

## 5. 对 Agent 架构的认识

这次实验进一步明确了 **Tool 和 LLM 的职责边界**：

| 组件          | 主要职责                                     |
| ------------- | -------------------------------------------- |
| LLM           | 理解用户意图、决定是否调用工具、组织最终回答 |
| Tool          | 执行确定性的外部操作或计算                   |
| Tool Result   | 给 LLM 提供外部事实/执行结果                 |
| Agent Runtime | 管理 LLM → Tool → LLM 的循环                 |

因此：

```
             User
               ↓
              LLM
               ↓
         Tool Call 请求
               ↓
        Agent Runtime
               ↓
             Tool
               ↓
          Tool Result
               ↓
              LLM
               ↓
          Final Answer
```

其中最后一次 LLM 调用依然是一个**生成过程**，而不是简单的结果转发。

------

### 6. 一个值得记录的设计原则

这次实验还说明：

> **如果某个事实非常重要，就应该让 Tool 返回完整、明确的数据，而不是期待 LLM 自己计算。**

例如不要只返回：

```
True
```

而可以返回：

```
{
    "is_fibonacci": True,
    "index": 13
}
```

这样 LLM 的职责就更加明确：

```
Tool
负责事实

LLM
负责解释事实
```

这也是后续设计 Agent 工具时需要考虑的问题：**Tool 返回什么信息，会直接影响 Agent 后续能可靠地做什么。**