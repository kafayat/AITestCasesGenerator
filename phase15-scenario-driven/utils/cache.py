"""
Simple file-based cache so identical generation requests (same model,
same requirement, same analysis, same RAG context, same coverage group)
never have to re-hit the LLM. This matters a lot while iterating on
prompts/UI during development and demos, where the same requirement gets
re-run repeatedly.

Cache is intentionally simple (one JSON file per key) so it is easy to
inspect/delete by hand. It is NOT meant to be a production cache - it is
a development-speed and demo-speed optimization.
"""

import hashlib
import json
import os
import time


CACHE_DIR = "cache"


def _ensure_cache_dir(cache_dir=CACHE_DIR):
    os.makedirs(cache_dir, exist_ok=True)


def make_cache_key(*parts):
    """
    Build a stable hash key from any number of string parts
    (e.g. model name, task name, requirement text, rag context, group id).
    """

    hasher = hashlib.sha256()

    for part in parts:
        hasher.update(str(part).encode("utf-8"))
        hasher.update(b"\x00")

    return hasher.hexdigest()


def get_cached(key, cache_dir=CACHE_DIR):

    _ensure_cache_dir(cache_dir)

    path = os.path.join(cache_dir, f"{key}.json")

    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        return payload.get("value")
    except Exception:
        return None


def set_cached(key, value, cache_dir=CACHE_DIR):

    _ensure_cache_dir(cache_dir)

    path = os.path.join(cache_dir, f"{key}.json")

    payload = {
        "cached_at": time.time(),
        "value": value,
    }

    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
    except Exception:
        # Caching is a speed optimization, not a correctness requirement.
        # Never let a cache write failure break generation.
        pass


def clear_cache(cache_dir=CACHE_DIR):

    if not os.path.exists(cache_dir):
        return

    for file_name in os.listdir(cache_dir):
        if file_name.endswith(".json"):
            try:
                os.remove(os.path.join(cache_dir, file_name))
            except Exception:
                pass
