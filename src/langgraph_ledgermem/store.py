"""LangGraph ``BaseStore`` implementation backed by Mnemo.

LangGraph stores are addressed by ``(namespace, key)`` and hold arbitrary JSON
values. We map this onto Mnemo by:

* Encoding the value as a JSON string in the memory ``content``.
* Persisting ``namespace`` and ``key`` in the memory ``metadata`` so the same
  pair can be resolved later.

Vector search over stored values is delegated to ``Mnemo.search`` via the
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
from getmnemo import Mnemo


def _ns_to_str(namespace: tuple[str, ...]) -> str:
    return "/".join(namespace)


def _str_to_ns(raw: str) -> tuple[str, ...]:
    return tuple(part for part in raw.split("/") if part)


class MnemoStore(BaseStore):
    """A LangGraph long-term store with Mnemo as the backing memory layer."""

    def __init__(self, client: Mnemo) -> None:
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
        # Prefer real timestamps from the memory record; fall back to "now"
        # only if the backend didn't supply them. Stamping every hit with
        # the current time made created_at/updated_at useless for ordering
        # and for incremental sync.
        now = datetime.now(timezone.utc)
        created_at = (
            getattr(hit, "created_at", None)
            or meta.pop("created_at", None)
            or now
        )
        updated_at = (
            getattr(hit, "updated_at", None)
            or meta.pop("updated_at", None)
            or created_at
        )
        return Item(
            value=value,
            key=key,
            namespace=ns,
            created_at=created_at,
            updated_at=updated_at,
        )

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
        # Walk the same listing we used to find the memory id and pull the
        # record out directly. The previous implementation re-fetched via
        # semantic search using op.key as the query, which silently returned
        # None whenever the matching row wasn't in the top-10 search hits —
        # i.e. for any non-trivial workspace the get() round-tripped to
        # 'not found' even though _find_memory_id had just confirmed it.
        ns_str = _ns_to_str(op.namespace)
        cursor: str | None = None
        while True:
            page = self._client.list(limit=100, cursor=cursor)
            items = getattr(page, "items", []) or getattr(page, "memories", []) or []
            for item in items:
                meta = getattr(item, "metadata", {}) or {}
                if meta.get("ns") == ns_str and meta.get("key") == op.key:
                    return self._hit_to_item(item)
            cursor = getattr(page, "next_cursor", None)
            if not cursor:
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
        # op.limit may legitimately be 0; only fall back when None.
        limit = 10 if op.limit is None else op.limit
        # Over-fetch so namespace/filter pruning still leaves `limit` items.
        # Without this, a large workspace can return zero matches inside the
        # requested namespace prefix even when many exist.
        fetch_limit = max(limit * 5, 50) if (op.namespace_prefix or op.filter) else limit
        response = self._client.search(query, limit=fetch_limit)
        ns_prefix_tuple = tuple(op.namespace_prefix or ())
        op_filter = op.filter or {}
        out: list[SearchItem] = []
        for hit in getattr(response, "hits", []) or []:
            meta = getattr(hit, "metadata", {}) or {}
            # Compare namespaces tuple-wise. The previous string-startswith
            # check matched ``ns_prefix=("user",)`` against stored
            # ``ns="users/123"`` because "users/123".startswith("user") is
            # True — leaking memories from sibling namespaces whose first
            # segment merely shared a string prefix.
            hit_ns = _str_to_ns(str(meta.get("ns", "")))
            if ns_prefix_tuple and hit_ns[: len(ns_prefix_tuple)] != ns_prefix_tuple:
                continue
            if op_filter:
                base_value_for_filter = None
                # SearchOp.filter applies to stored values; load lazily.
                base = self._hit_to_item(hit)
                base_value_for_filter = base.value if isinstance(base.value, dict) else None
                if base_value_for_filter is None:
                    continue
                if not all(base_value_for_filter.get(k) == v for k, v in op_filter.items()):
                    continue
            else:
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
            if len(out) >= limit:
                break
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
        # Honour ListNamespacesOp filters — without these the caller gets
        # the entire workspace's namespace tree even when they asked for a
        # specific prefix/suffix or a bounded depth. Both fields are
        # tuple[MatchCondition, ...] in newer LangGraph and a plain prefix
        # tuple in older versions; handle both shapes.
        match_conditions = getattr(op, "match_conditions", None) or ()
        for cond in match_conditions:
            match_type = getattr(cond, "match_type", None)
            path = tuple(getattr(cond, "path", ()) or ())
            if not path:
                continue
            if match_type == "prefix":
                seen = {ns for ns in seen if ns[: len(path)] == path}
            elif match_type == "suffix":
                seen = {ns for ns in seen if ns[-len(path) :] == path}
        max_depth = getattr(op, "max_depth", None)
        if isinstance(max_depth, int) and max_depth > 0:
            seen = {ns[:max_depth] for ns in seen}
        return sorted(seen)
