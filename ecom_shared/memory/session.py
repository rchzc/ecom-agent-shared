"""会话记忆（短期）：按 session_id 维护多轮对话上下文。

**它是"多轮指代解析"的地基。** 用户说"它有什么优惠"时，模型必须能从历史里
定位到上一轮聊的是哪个商品 —— 只看当前这句话，宾语是缺失的。

三个设计点：

1. **按轮数截断，不按字符数。** 截断的目的是控制 token 成本，而
   "保留最近 N 轮"比"保留最近 N 个字符"更符合对话的语义结构 ——
   按字符截会把最后一轮用户提问拦腰砍掉，模型收到半句话，回答直接跑偏。
2. **user 和 assistant 成对保留。** 只留 user 不留 assistant，模型看不到自己
   上一轮说了什么，会导致重复回答或前后矛盾。所以按 2×max_turns 条截断。
3. **进程内存储，不引 Redis。** 诚实说明这是当前规模的取舍：
   单实例部署够用，多实例或需要重启保留就必须换 Redis。
   接口已经收敛成 append / history / clear 三个方法，
   换成 Redis 实现不影响任何调用方（README「已知边界」里写了这条）。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Turn:
    role: str
    content: str


@dataclass
class SessionMemory:
    """进程内的多轮会话记忆。"""

    max_turns: int = 10
    #: 超出上限时丢弃最旧的对话。用 defaultdict(list) 免去每次 setdefault
    _store: dict[str, list[Turn]] = field(default_factory=lambda: defaultdict(list))

    def append(self, session_id: str, role: str, content: str) -> None:
        if not session_id:
            raise ValueError("session_id 不能为空")
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"非法 role: {role!r}")
        history = self._store[session_id]
        history.append(Turn(role=role, content=content))
        # user / assistant 成对，所以上限是 max_turns * 2
        limit = self.max_turns * 2
        if len(history) > limit:
            del history[: len(history) - limit]

    def extend(self, session_id: str, turns: Iterable[Turn]) -> None:
        for turn in turns:
            self.append(session_id, turn.role, turn.content)

    def history(self, session_id: str) -> list[Turn]:
        return list(self._store.get(session_id, []))

    def messages(self, session_id: str) -> list[dict[str, str]]:
        """转成 OpenAI 兼容的 messages 格式，可直接拼进请求。"""
        return [{"role": t.role, "content": t.content} for t in self.history(session_id)]

    def last_user_query(self, session_id: str) -> str:
        """最近一条用户输入。用于指代解析与检索 query 构造。"""
        for turn in reversed(self._store.get(session_id, [])):
            if turn.role == "user":
                return turn.content
        return ""

    def last_assistant_reply(self, session_id: str) -> str:
        for turn in reversed(self._store.get(session_id, [])):
            if turn.role == "assistant":
                return turn.content
        return ""

    def transcript(self, session_id: str) -> str:
        """拼成纯文本转录。喂给"评分 / 复盘"类任务（如销售考核）比 messages 更好用，
        因为那类任务要的是"整段对话"而不是"对话列表"。"""
        label = {"user": "客户", "assistant": "销售", "system": "系统"}
        return "\n".join(
            f"{label.get(t.role, t.role)}：{t.content}"
            for t in self._store.get(session_id, [])
        )

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def sessions(self) -> list[str]:
        return list(self._store.keys())

    def size(self) -> int:
        """当前会话数与总轮数 —— 用于前端展示内存占用，避免无界增长无人察觉。"""
        return sum(len(v) for v in self._store.values())


if __name__ == "__main__":
    mem = SessionMemory(max_turns=3)
    mem.append("s1", "user", "推荐一款降噪耳机")
    mem.append("s1", "assistant", "推荐 A 款主动降噪耳机")
    mem.append("s1", "user", "它有什么优惠")
    assert mem.last_user_query("s1") == "它有什么优惠"
    assert len(mem.history("s1")) == 3
    # 超限后最旧的被丢掉，且保持"最近一轮完整"
    for i in range(10):
        mem.append("s1", "user", f"追问{i}")
        mem.append("s1", "assistant", f"回答{i}")
    assert len(mem.history("s1")) == 6, len(mem.history("s1"))
    assert mem.last_user_query("s1") == "追问9"
    print("SessionMemory 自测通过：", mem.transcript("s1").splitlines()[-1][:20])
