"""Small HTTP helpers shared by the ingesters (stdlib only)."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path


def get(url: str, retries: int = 3, timeout: int = 120) -> bytes:
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404 or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def etag(url: str) -> str | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.headers.get("ETag", "").strip('"') or r.headers.get("Last-Modified")
    except urllib.error.URLError:
        return None


def cached(url: str, cache_dir: Path) -> Path:
    dest = cache_dir / url.split("://", 1)[1]
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(get(url))
        tmp.replace(dest)
    return dest
