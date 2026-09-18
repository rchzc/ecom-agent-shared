"""长期记忆：跨会话保留的用户事实与偏好。

和短期会话记忆的区别：会话记忆回答"刚才聊了什么"，长期记忆回答
"这个客户三个月前说过他主要做北美站、对价格敏感"。

**为什么不用向量检索（这是一个有意的取舍）：**

长期记忆的条目是短句（"客户主营北美站"），条数量级是几百到几千，
远达不到需要 ANN 索引的规模。这个规模下，一次 embedding 调用（网络往返 100ms+）
换来的收益，还不如直接做词面匹配 —— 而且 embedding 调用本身就是链路里最容易挂的一环。

所以这里用 **词面重叠 × 时间衰减** 打分：

    score = overlap(query, fact) × decay(age)

时间衰减用指数函数，半衰期可配。理由是事实会过期 ——
"客户上季度主推 A 商品"这种事实，半年后检索出来反而是误导。
不用简单的"只看最近 N 条"，因为真正重要的事实（如"客户不接受预售"）
不该因为最近聊了别的就检索不到。

**存储是可插拔的**：默认进程内，实现 `MemoryStore` 协议的都能接
（Redis / SQLite / 向量库）。接口只有 read/write 两个方法。
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Iterable, Protocol

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[一-鿿]")

#: 时间衰减半衰期（天）。14 天意味着两周前的记忆权重降到一半。
DEFAULT_HALF_LIFE_DAYS = 14.0


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


@dataclass
class Fact:
    """一条长期记忆。"""

    key: str
    value: str
    #: 事实所属的命名空间，通常是 user_id / 店铺 ID / 会话主题
    scope: str = "default"
    #: Unix 时间戳（秒）。写入时自动填，显式传入是为了支持从历史数据导入
    created_at: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return f"{self.key}：{self.value}"

    def age_days(self, now: float | None = None) -> float:
        return max((now or time.time()) - self.created_at, 0.0) / 86400.0

    def decay(self, half_life_days: float = DEFAULT_HALF_LIFE_DAYS, now: float | None = None) -> float:
        """指数时间衰减，返回 0~1。刚写入时为 1，每过一个半衰期减半。"""
        if half_life_days <= 0:
            return 1.0
        return 0.5 ** (self.age_days(now) / half_life_days)

    def score(self, query: str, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> float:
        """词面重叠 × 时间衰减。query 无有效 token 时退化为纯时间排序。"""
        q = _tokens(query)
        if not q:
            return self.decay(half_life_days)
        overlap = len(q & _tokens(self.text)) / len(q)
        return round(overlap * self.decay(half_life_days), 4)


class MemoryStore(Protocol):
    """存储后端契约。换成 Redis / SQLite 只需实现这两个方法。"""

    def load(self, scope: str) -> list[Fact]: ...

    def save(self, scope: str, facts: list[Fact]) -> None: ...


class InMemoryStore:
    """默认存储：进程内字典。重启即丢，适合演示与单实例。"""

    def __init__(self) -> None:
        self._data: dict[str, list[Fact]] = {}

    def load(self, scope: str) -> list[Fact]:
        return list(self._data.get(scope, []))

    def save(self, scope: str, facts: list[Fact]) -> None:
        self._data[scope] = list(facts)


class JsonFileStore:
    """把记忆落到 JSON 文件。比内存版多一层"重启不丢"，实现仍然只有几行。

    没有直接上 SQLite 是因为这里的访问模式极其简单（按 scope 整块读写），
    用不上事务和查询能力，加了反而是过度设计。
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._cache: dict[str, list[Fact]] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            self._cache = {
                scope: [Fact(**item) for item in items] for scope, items in raw.items()
            }

    def load(self, scope: str) -> list[Fact]:
        return list(self._cache.get(scope, []))

    def save(self, scope: str, facts: list[Fact]) -> None:
        self._cache[scope] = list(facts)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(
                {s: [asdict(f) for f in items] for s, items in self._cache.items()},
                fh,
                ensure_ascii=False,
                indent=2,
            )


class LongTermMemory:
    """长期事实库。写入去重、检索按相关性排序。"""

    def __init__(
        self,
        store: MemoryStore | None = None,
        *,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        max_facts_per_scope: int = 200,
    ) -> None:
        self.store = store or InMemoryStore()
        self.half_life_days = half_life_days
        self.max_facts_per_scope = max_facts_per_scope

    def remember(
        self,
        key: str,
        value: str,
        *,
        scope: str = "default",
        tags: Iterable[str] = (),
        created_at: float | None = None,
    ) -> None:
        """写入一条事实。同 key 覆盖而不追加 —— 事实会更新，
        留着"客户主营北美站"和"客户主营欧洲站"两条只会互相矛盾。
        """
        facts = [f for f in self.store.load(scope) if f.key != key]
        facts.append(
            Fact(
                key=key,
                value=value,
                scope=scope,
                tags=list(tags),
                created_at=created_at if created_at is not None else time.time(),
            )
        )
        # 超出上限时丢弃最早写入的，防止单用户把内存撑爆
        if len(facts) > self.max_facts_per_scope:
            facts.sort(key=lambda f: f.created_at)
            facts = facts[-self.max_facts_per_scope :]
        self.store.save(scope, facts)

    def recall(
        self, query: str, *, scope: str = "default", top_k: int = 5, min_score: float = 0.0
    ) -> list[Fact]:
        """按相关性取回事实。"""
        facts = self.store.load(scope)
        if not facts:
            return []
        scored = [(f.score(query, self.half_life_days), f) for f in facts]
        scored = [item for item in scored if item[0] > min_score or not query.strip()]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [f for _, f in scored[:top_k]]

    def all(self, scope: str = "default") -> list[Fact]:
        return sorted(self.store.load(scope), key=lambda f: f.created_at, reverse=True)

    def forget(self, key: str, *, scope: str = "default") -> bool:
        facts = self.store.load(scope)
        kept = [f for f in facts if f.key != key]
        if len(kept) == len(facts):
            return False
        self.store.save(scope, kept)
        return True

    def purge(self, scope: str = "default") -> None:
        self.store.save(scope, [])

    def as_prompt_block(self, query: str, *, scope: str = "default", top_k: int = 5) -> str:
        """拼成可直接插进 system prompt 的一段。

        带 `[记忆]` 前缀是**刻意的**：模型看到带标记的段落会更谨慎地使用它，
        而且出问题时能一眼看出这段来自记忆而非知识库检索 ——
        "回答错了"和"记忆里存了过期的错事实"是两种完全不同的故障。
        """
        facts = self.recall(query, scope=scope, top_k=top_k)
        if not facts:
            return ""
        lines = "\n".join(f"- {f.text}" for f in facts)
        return f"[记忆] 以下是关于该客户的已知事实，回答时请结合使用：\n{lines}"


if __name__ == "__main__":
    mem = LongTermMemory(half_life_days=14)
    now = time.time()
    mem.remember("主营站点", "北美站（US + CA）", scope="u1")
    mem.remember("价格敏感度", "对价格敏感，常用优惠券", scope="u1")
    # 写一条半年前的事实，验证时间衰减确实把它压下去
    mem.remember("上季主推", "A 款蓝牙耳机", scope="u1", created_at=now - 180 * 86400)

    hits = mem.recall("北美站 价格", scope="u1", top_k=2)
    assert hits and "北美" in hits[0].text, [f.text for f in hits]
    assert all("上季主推" not in h.text for h in hits), "过期事实不该排在前面"
    # 同 key 覆盖，不追加
    mem.remember("主营站点", "欧洲站（DE + FR）", scope="u1")
    assert len(mem.all("u1")) == 3, mem.all("u1")
    print("LongTermMemory 自测通过：", mem.as_prompt_block("价格", scope="u1").replace("\n", " | "))
