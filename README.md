# ledgermem-langgraph

LangGraph `BaseStore` backed by [LedgerMem](https://github.com/ledgermem/ledgermem-python) — give your agents persistent, searchable cross-thread memory in one line.

## Install

```bash
pip install ledgermem-langgraph
```

## Quickstart

```python
from langgraph.graph import StateGraph, MessagesState
from ledgermem import LedgerMem
from langgraph_ledgermem import LedgerMemStore

store = LedgerMemStore(LedgerMem(api_key="lm_...", workspace_id="ws_..."))

# Write
store.put(("users", "u1"), "profile", {"name": "Ada", "tz": "PKT"})

# Read by exact key
item = store.get(("users", "u1"), "profile")
print(item.value)

# Semantic search across the namespace
hits = store.search(("users",), query="who lives in Pakistan?", limit=5)
for hit in hits:
    print(hit.score, hit.value)
```

## Wire into a LangGraph workflow

```python
graph = StateGraph(MessagesState)
# ... add nodes / edges ...
app = graph.compile(store=store)
```

The store is passed to every node via the standard `config["configurable"]["store"]` channel — your nodes can `store.put(...)` and `store.search(...)` to share long-term context across threads.

## License

MIT — see [LICENSE](./LICENSE).
