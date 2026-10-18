"""mverify: verify model output of any format against a database of cleaned observations."""

__version__ = "0.1.0.dev0"


def __getattr__(name):  # keep `import mverify` light; load the heavy modules on first use
    if name == "open_forecast":
        from mverify.io.open import open_forecast
        return open_forecast
    if name == "verify":
        from mverify.verify import verify
        return verify
    if name == "ObsStore":
        from mverify.db.store import ObsStore
        return ObsStore
    raise AttributeError(name)
