# betcomb/cache.py
import os, time, json, hashlib, pickle
from typing import Any, Optional

try:
    # Tu config habitual
    from .config import SETTINGS
    CACHE_DIR = getattr(SETTINGS, "cache_dir", ".betcomb_cache")
except Exception:
    # Fallback si SETTINGS no tiene cache_dir
    CACHE_DIR = ".betcomb_cache"

def _ensure_cache_dir() -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR

def make_key(kind: str, provider: str, items: list, days: int, extra: str = "") -> str:
    """
    Genera una clave estable (hash) a partir de los parámetros para diferenciar
    archivos de cache por tipo de dato, proveedor, lista de items y ventana en días.
    """
    payload = {"kind": kind, "provider": provider, "items": items, "days": days, "extra": extra}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return f"{kind}_{provider}_{hashlib.md5(raw.encode('utf-8')).hexdigest()}"

def _path_for(key: str) -> str:
    base = _ensure_cache_dir()
    return os.path.join(base, f"{key}.pkl")

def load_pickle(key: str, max_age_s: Optional[int] = None) -> Optional[Any]:
    """
    Lee y devuelve el objeto cacheado si existe y no está vencido.
    Si no hay archivo o está vencido, devuelve None.
    """
    p = _path_for(key)
    if not os.path.exists(p):
        return None
    if max_age_s is not None:
        age = time.time() - os.path.getmtime(p)
        if age > max_age_s:
            return None
    with open(p, "rb") as f:
        return pickle.load(f)

def save_pickle(key: str, obj: Any) -> str:
    """
    Guarda el objeto en disco como pickle y devuelve la ruta.
    """
    p = _path_for(key)
    with open(p, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return p
