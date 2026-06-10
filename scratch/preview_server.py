"""Local preview launcher: stubs heavy ML deps (absent in this container),
then serves the dashboard. Not committed-critical; for previewing the UI."""
import types, sys

def _stub(name, attrs=None, submodules=None):
    import importlib.util
    if importlib.util.find_spec(name) is not None or name in sys.modules:
        return
    mod = types.ModuleType(name); mod.__path__ = []
    for k, v in (attrs or {}).items(): setattr(mod, k, v)
    sys.modules[name] = mod
    for sub, sattrs in (submodules or {}).items():
        full = f"{name}.{sub}"; s = types.ModuleType(full)
        for k, v in (sattrs or {}).items(): setattr(s, k, v)
        sys.modules[full] = s; setattr(mod, sub, s)

_stub("sentence_transformers", {"SentenceTransformer": object, "CrossEncoder": object},
      {"util": {"cos_sim": lambda *a, **k: None}})
_stub("faiss")
_stub("rank_bm25", {"BM25Okapi": object})

import uvicorn
from app.db.init_db import init_db
init_db()
uvicorn.run("app.api.server:app", host="127.0.0.1", port=8000, log_level="warning")
