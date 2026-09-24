from typing import Callable, Optional

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
)


class ContextManager:

    def __init__(
        self,
        max_tokens=100,
        keep_recent=3,
        summarizer: Optional[Callable[[str, str], str]] = None,
    ):
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self.summarizer = summarizer

        # 压缩后的长期摘要
        self.summary = ""

        # 尚未被压缩的短期历史
        self.history = []

    # ============================================================
    # History
    # ============================================================

    def add_message(self, message):
        """向短期历史中添加一条消息。"""
        self.history.append(message)

    # ============================================================
    # Context
    # ============================================================

    def get_messages(self):
        """
        获取当前应该发送给 LLM 的上下文。

        注意：
        这里只读取数据，不进行压缩。
        """

        messages = []

        # 如果存在历史摘要，把摘要放在最前面
        if self.summary:
            messages.append(
                HumanMessage(
                    content=f"[历史摘要]\n{self.summary}"
                )
            )

        # 加入最近的原始历史
        messages.extend(self.history)

        return messages

    # ============================================================
    # Compression
    # ============================================================

    def should_compress(self):
        """
        判断当前短期历史是否需要压缩。
        """

        return (
            len(self.history) > self.keep_recent
            and self._count_tokens(self.history) > self.max_tokens
        )

    def compress(self):
        """
        将旧的短期历史压缩成新的 Summary。

        注意：
        Summary 是覆盖更新，而不是不断追加。
        """

        if not self.should_compress():
            return

        # --------------------------------------------------------
        # 计算需要被压缩的消息数量
        # --------------------------------------------------------

        cut = len(self.history) - self.keep_recent

        # 防止从 ToolMessage 中间切断
        cut = self._safe_cut(cut)

        if cut <= 0:
            return

        # --------------------------------------------------------
        # 分离：
        #
        # old      → 需要压缩
        # history  → 保留最近消息
        # --------------------------------------------------------

        old = self.history[:cut]
        self.history = self.history[cut:]

        # --------------------------------------------------------
        # 转换成适合摘要模型阅读的文本
        # --------------------------------------------------------

        old_text = self._render(old)

        # --------------------------------------------------------
        # 调用摘要器
        # --------------------------------------------------------

        if self.summarizer is not None:

            new_summary = self.summarizer(
                self.summary,
                old_text,
            )

        else:

            # 没有 summarizer 时直接保存文本
            new_summary = old_text

        # --------------------------------------------------------
        # 关键：
        #
        # 覆盖旧 Summary
        #
        # 而不是：
        #
        # self.summary += new_summary
        # --------------------------------------------------------

        self.summary = new_summary.strip()

    # ============================================================
    # Token Estimate
    # ============================================================

    def _count_tokens(self, messages) -> int:
        """
        粗略估算 Token 数。

        当前实验阶段：
        约 4 个字符 ≈ 1 token。

        注意：
        这不是实际 tokenizer 结果。
        """

        total = 0

        for message in messages:

            content = (
                message.content
                if isinstance(message.content, str)
                else str(message.content)
            )

            total += len(content)

            # AIMessage 中的 tool_call 也占一定上下文空间
            if isinstance(message, AIMessage):

                tool_calls = getattr(
                    message,
                    "tool_calls",
                    []
                )

                total += 50 * len(tool_calls)

            # 粗略估计 message metadata / role 等额外开销
            total += 4

        return total // 4

    # ============================================================
    # Safe Cut
    # ============================================================

    def _safe_cut(self, cut: int) -> int:
        """
        防止在 ToolMessage 上进行裁剪。

        例如：

        AIMessage
            ↓
        ToolMessage

        不能只留下 ToolMessage。
        """

        while (
            cut > 0
            and isinstance(
                self.history[cut],
                ToolMessage
            )
        ):
            cut -= 1

        return cut

    # ============================================================
    # Render
    # ============================================================

    def _render(self, messages) -> str:
        """
        将 LangChain Message 转换成纯文本，
        交给 Summary LLM。
        """

        lines = []

        for message in messages:

            # ----------------------------------------------------
            # Role
            # ----------------------------------------------------

            if isinstance(message, HumanMessage):

                role = "User"

            elif isinstance(message, AIMessage):

                role = "Assistant"

            elif isinstance(message, ToolMessage):

                role = f"Tool({message.name})"

            else:

                role = type(message).__name__

            # ----------------------------------------------------
            # Content
            # ----------------------------------------------------

            content = (
                message.content
                if isinstance(message.content, str)
                else str(message.content)
            )

            # ----------------------------------------------------
            # Tool Call
            # ----------------------------------------------------

            if isinstance(message, AIMessage):

                tool_calls = getattr(
                    message,
                    "tool_calls",
                    []
                )

                if tool_calls:

                    names = []

                    for tool_call in tool_calls:

                        if isinstance(tool_call, dict):

                            names.append(
                                tool_call.get(
                                    "name",
                                    "?"
                                )
                            )

                        else:

                            names.append(
                                getattr(
                                    tool_call,
                                    "name",
                                    "?"
                                )
                            )

                    content += (
                        f" [调用工具: "
                        f"{', '.join(names)}]"
                    )

            lines.append(
                f"{role}: {content}"
            )

        return "\n".join(lines)