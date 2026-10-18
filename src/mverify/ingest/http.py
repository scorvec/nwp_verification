"""Small HTTP helpers shared by the ingesters (stdlib only)."""

from __future__ import annotations

import http.client
import http.cookiejar
import netrc
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

EARTHDATA = "urs.earthdata.nasa.gov"


def earthdata_opener() -> urllib.request.OpenerDirector:
    """An opener that logs in to NASA Earthdata when a data server redirects to it.

    Credentials come from $EARTHDATA_USERNAME / $EARTHDATA_PASSWORD, else ~/.netrc
    (``machine urs.earthdata.nasa.gov login ... password ...``). The cookie jar keeps
    the session across the redirect back to the data server.
    """
    # Pasted secrets often carry a trailing newline, which Earthdata rejects as a wrong password.
    user = (os.environ.get("EARTHDATA_USERNAME") or "").strip()
    password = (os.environ.get("EARTHDATA_PASSWORD") or "").strip()
    if not (user and password):
        try:
            auth = netrc.netrc().authenticators(EARTHDATA)
        except (FileNotFoundError, netrc.NetrcParseError):
            auth = None
        if auth is None:
            raise PermissionError("NASA Earthdata credentials needed: set EARTHDATA_USERNAME and "
                                  f"EARTHDATA_PASSWORD, or add {EARTHDATA} to ~/.netrc")
        user, password = auth[0], auth[2]
    # Sent with the first request to Earthdata rather than after its 401 challenge, which does not
    # always come back in a form urllib retries.
    pm = urllib.request.HTTPPasswordMgrWithPriorAuth()
    pm.add_password(None, f"https://{EARTHDATA}", user, password, is_authenticated=True)
    return urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(pm),
                                       urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))


def get(url: str, retries: int = 3, timeout: int = 120,
        opener: urllib.request.OpenerDirector | None = None) -> bytes:
    for attempt in range(retries + 1):
        try:
            with (opener.open(url, timeout=timeout) if opener else urllib.request.urlopen(url, timeout=timeout)) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 401 and opener is not None:
                # Say why: Earthdata answers a wrong login and an unapproved data application alike.
                detail = re.sub(r"<[^>]+>|\s+", " ", e.read().decode(errors="replace")).strip()[:300]
                raise PermissionError(f"NASA Earthdata refused the login for {url} ({detail or e.reason}). "
                                      "Check the username/password, and that the data centre's application "
                                      "(e.g. 'NASA GESDISC DATA ARCHIVE') is approved in the Earthdata "
                                      "profile under Applications > Authorized Apps.") from e
            if e.code == 404 or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead):
            # IncompleteRead: the server closed the connection mid-body (seen on the IEM archive)
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


def cached(url: str, cache_dir: Path, opener: urllib.request.OpenerDirector | None = None) -> Path:
    dest = cache_dir / url.split("://", 1)[1]
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(get(url, opener=opener))
        tmp.replace(dest)
    return dest
