"""Smoke test: import + instantiate the LangGraph store with a mocked SDK."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _install_fake_ledgermem() -> None:
    if "ledgermem" in sys.modules:
        return
    fake = types.ModuleType("ledgermem")

    class LedgerMem:
        def __init__(self, *a, **k):
            pass

        def search(self, query, limit=5):
            return types.SimpleNamespace(hits=[])

        def add(self, content, metadata=None):
            return types.SimpleNamespace(id="mem_test")

        def update(self, memory_id, **kwargs):
            return types.SimpleNamespace(id=memory_id)

        def delete(self, memory_id):
            return None

        def list(self, limit=20, cursor=None):
            return types.SimpleNamespace(items=[], next_cursor=None)

    class AsyncLedgerMem(LedgerMem):
        pass

    fake.LedgerMem = LedgerMem
    fake.AsyncLedgerMem = AsyncLedgerMem
    sys.modules["ledgermem"] = fake


_install_fake_ledgermem()

from langgraph_ledgermem import LedgerMemStore  # noqa: E402
from ledgermem import LedgerMem  # noqa: E402


def test_store_imports() -> None:
    assert LedgerMemStore is not None


def test_put_writes_to_sdk() -> None:
    client = LedgerMem()
    client.add = MagicMock(return_value=None)
    client.list = MagicMock(return_value=type("P", (), {"items": [], "next_cursor": None})())
    store = LedgerMemStore(client)
    store.put(("users", "u1"), "profile", {"name": "Ada"})
    assert client.add.called
    args, kwargs = client.add.call_args
    assert "Ada" in args[0]
    assert kwargs["metadata"]["ns"] == "users/u1"
    assert kwargs["metadata"]["key"] == "profile"


def test_search_filters_by_namespace_prefix() -> None:
    client = LedgerMem()
    hit = type(
        "Hit",
        (),
        {"id": "m1", "content": '{"name": "Ada"}', "metadata": {"ns": "users/u1", "key": "profile"}, "score": 0.8},
    )()
    client.search = MagicMock(return_value=type("R", (), {"hits": [hit]})())
    store = LedgerMemStore(client)
    items = store.search(("users",), query="Ada")
    assert len(items) == 1
    assert items[0].value == {"name": "Ada"}
