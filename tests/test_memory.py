"""记忆层单测：短期会话记忆 + 长期事实记忆。"""
from __future__ import annotations

import time

import pytest

from ecom_shared.memory.longterm import InMemoryStore, JsonFileStore, LongTermMemory
from ecom_shared.memory.session import SessionMemory


# ---------------------------------------------------------------------------
# 会话记忆
# ---------------------------------------------------------------------------
def test_session_keeps_recent_turns_and_drops_oldest():
    mem = SessionMemory(max_turns=2)
    for i in range(5):
        mem.append("s", "user", f"问{i}")
        mem.append("s", "assistant", f"答{i}")
    history = mem.history("s")
    assert len(history) == 4  # max_turns * 2
    assert history[-1].content == "答4"
    assert history[0].content == "问3"


def test_session_truncation_keeps_pairs_not_half_turns():
    """按轮截断而不是按字符截 —— 截到半句话会让模型收到残缺提问。"""
    mem = SessionMemory(max_turns=1)
    mem.append("s", "user", "很长的问题" * 100)
    mem.append("s", "assistant", "回答")
    mem.append("s", "user", "新问题")
    mem.append("s", "assistant", "新回答")
    history = mem.history("s")
    assert history[0].role == "user"
    assert history[-1].role == "assistant"


def test_session_last_user_query_skips_assistant():
    mem = SessionMemory()
    mem.append("s", "user", "推荐一款降噪耳机")
    mem.append("s", "assistant", "推荐 A 款")
    assert mem.last_user_query("s") == "推荐一款降噪耳机"


def test_session_transcript_labels_roles():
    mem = SessionMemory()
    mem.append("s", "user", "太贵了")
    mem.append("s", "assistant", "有优惠券")
    text = mem.transcript("s")
    assert "客户：太贵了" in text and "销售：有优惠券" in text


def test_session_rejects_bad_role_and_empty_id():
    mem = SessionMemory()
    with pytest.raises(ValueError):
        mem.append("s", "robot", "x")
    with pytest.raises(ValueError):
        mem.append("", "user", "x")


def test_session_isolated_by_session_id():
    mem = SessionMemory()
    mem.append("a", "user", "A 的问题")
    mem.append("b", "user", "B 的问题")
    assert mem.last_user_query("a") == "A 的问题"
    assert mem.last_user_query("b") == "B 的问题"
    mem.clear("a")
    assert mem.history("a") == []


# ---------------------------------------------------------------------------
# 长期记忆
# ---------------------------------------------------------------------------
def test_longterm_same_key_overwrites_instead_of_appending():
    """事实会更新。留着"主营北美站"和"主营欧洲站"两条只会互相矛盾。"""
    mem = LongTermMemory()
    mem.remember("主营站点", "北美站", scope="u1")
    mem.remember("主营站点", "欧洲站", scope="u1")
    facts = mem.all("u1")
    assert len(facts) == 1
    assert facts[0].value == "欧洲站"


def test_longterm_time_decay_pushes_stale_facts_down():
    mem = LongTermMemory(half_life_days=14)
    now = time.time()
    mem.remember("主营站点", "北美站", scope="u1", created_at=now)
    # 半年前的一条事实，即便字面高度匹配也不该排第一
    mem.remember("主营站点历史", "北美站老资料", scope="u1", created_at=now - 180 * 86400)

    hits = mem.recall("主营站点", scope="u1", top_k=2)
    assert "历史" not in hits[0].key


def test_longterm_recall_empty_when_nothing_stored():
    assert LongTermMemory().recall("任意") == []


def test_longterm_forget_and_purge():
    mem = LongTermMemory()
    mem.remember("k", "v", scope="u1")
    assert mem.forget("k", scope="u1") is True
    assert mem.forget("nope", scope="u1") is False
    mem.remember("k2", "v", scope="u1")
    mem.purge("u1")
    assert mem.all("u1") == []


def test_longterm_scope_isolation():
    mem = LongTermMemory()
    mem.remember("k", "u1 的值", scope="u1")
    mem.remember("k", "u2 的值", scope="u2")
    assert mem.recall("k", scope="u1")[0].value == "u1 的值"


def test_longterm_prompt_block_marks_source():
    """带 [记忆] 前缀是刻意的：出问题时能一眼区分"记忆里存了过期事实"
    和"知识库检索错了"。"""
    mem = LongTermMemory()
    mem.remember("价格敏感度", "对价格敏感", scope="u1")
    block = mem.as_prompt_block("价格", scope="u1")
    assert block.startswith("[记忆]")
    assert "价格敏感度" in block


def test_longterm_caps_facts_per_scope():
    mem = LongTermMemory(max_facts_per_scope=3)
    for i in range(6):
        mem.remember(f"k{i}", f"v{i}", scope="u1", created_at=time.time() + i)
    assert len(mem.all("u1")) == 3


def test_json_file_store_survives_reload(tmp_path):
    path = str(tmp_path / "mem.json")
    mem = LongTermMemory(store=JsonFileStore(path))
    mem.remember("主营站点", "北美站", scope="u1")

    reloaded = LongTermMemory(store=JsonFileStore(path))
    assert reloaded.all("u1")[0].value == "北美站"


def test_in_memory_store_is_isolated_per_instance():
    a, b = InMemoryStore(), InMemoryStore()
    a.save("s", [])
    assert b.load("s") == []
