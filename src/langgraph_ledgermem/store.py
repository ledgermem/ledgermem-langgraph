"""LangGraph ``BaseStore`` implementation backed by LedgerMem.

LangGraph stores are addressed by ``(namespace, key)`` and hold arbitrary JSON
values. We map this onto LedgerMem by:

* Encoding the value as a JSON string in the memory ``content``.
* Persisting ``namespace`` and ``key`` in the memory ``metadata`` so the same
  pair can be resolved later.

Vector search over stored values is delegated to ``LedgerMem.search`` via the
optional ``query`` field on ``SearchOp``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterable

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)
from ledgermem import LedgerMem


def _ns_to_str(namespace: tuple[str, ...]) -> str:
    return "/".join(namespace)


def _str_to_ns(raw: str) -> tuple[str, ...]:
    return tuple(part for part in raw.split("/") if part)


class LedgerMemStore(BaseStore):
    """A LangGraph long-term store with LedgerMem as the backing memory layer."""

    def __init__(self, client: LedgerMem) -> None:
        self._client = client

    # --- helpers -----------------------------------------------------------

    def _find_memory_id(self, namespace: tuple[str, ...], key: str) -> str | None:
        ns_str = _ns_to_str(namespace)
        cursor: str | None = None
        while True:
            page = self._client.list(limit=100, cursor=cursor)
            items = getattr(page, "items", []) or getattr(page, "memories", []) or []
            for item in items:
                meta = getattr(item, "metadata", {}) or {}
                if meta.get("ns") == ns_str and meta.get("key") == key:
                    return getattr(item, "id", None)
            cursor = getattr(page, "next_cursor", None)
            if not cursor:
                return None

    def _hit_to_item(self, hit: Any) -> Item:
        meta = dict(getattr(hit, "metadata", {}) or {})
        ns = _str_to_ns(meta.pop("ns", ""))
        key = meta.pop("key", getattr(hit, "id", ""))
        raw = getattr(hit, "content", None) or getattr(hit, "text", "") or "{}"
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = {"text": raw}
        now = datetime.now(timezone.utc)
        return Item(value=value, key=key, namespace=ns, created_at=now, updated_at=now)

    # --- BaseStore API -----------------------------------------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._get(op))
            elif isinstance(op, PutOp):
                self._put(op)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(self._search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._list_namespaces(op))
            else:
                raise NotImplementedError(f"Unsupported op: {type(op).__name__}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return self.batch(ops)

    # --- op handlers -------------------------------------------------------

    def _get(self, op: GetOp) -> Item | None:
        memory_id = self._find_memory_id(op.namespace, op.key)
        if memory_id is None:
            return None
        # Use a search call with the key as query to fetch the record back.
        response = self._client.search(op.key, limit=10)
        for hit in getattr(response, "hits", []) or []:
            if getattr(hit, "id", None) == memory_id:
                return self._hit_to_item(hit)
        return None

    def _put(self, op: PutOp) -> None:
        ns_str = _ns_to_str(op.namespace)
        existing = self._find_memory_id(op.namespace, op.key)
        if op.value is None:
            if existing is not None:
                self._client.delete(existing)
            return
        content = json.dumps(op.value, default=str)
        metadata = {"ns": ns_str, "key": op.key, "source": "langgraph"}
        if existing is not None:
            self._client.update(existing, content=content, metadata=metadata)
        else:
            self._client.add(content, metadata=metadata)

    def _search(self, op: SearchOp) -> list[SearchItem]:
        query = op.query or ""
        limit = op.limit or 10
        response = self._client.search(query, limit=limit)
        ns_prefix = _ns_to_str(op.namespace_prefix)
        out: list[SearchItem] = []
        for hit in getattr(response, "hits", []) or []:
            meta = getattr(hit, "metadata", {}) or {}
            if ns_prefix and not str(meta.get("ns", "")).startswith(ns_prefix):
                continue
            base = self._hit_to_item(hit)
            out.append(
                SearchItem(
                    namespace=base.namespace,
                    key=base.key,
                    value=base.value,
                    created_at=base.created_at,
                    updated_at=base.updated_at,
                    score=getattr(hit, "score", None),
                )
            )
        return out

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        seen: set[tuple[str, ...]] = set()
        cursor: str | None = None
        while True:
            page = self._client.list(limit=100, cursor=cursor)
            items = getattr(page, "items", []) or getattr(page, "memories", []) or []
            for item in items:
                meta = getattr(item, "metadata", {}) or {}
                ns = _str_to_ns(meta.get("ns", ""))
                if ns:
                    seen.add(ns)
            cursor = getattr(page, "next_cursor", None)
            if not cursor:
                break
        return sorted(seen)
