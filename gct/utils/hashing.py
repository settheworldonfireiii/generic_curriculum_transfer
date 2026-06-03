from __future__ import annotations

import hashlib


def stable_id(*parts: object, prefix: str = "row") -> str:
    key = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()[:16]
    return f"{prefix}_{digest}"


def stable_seed(seed: int, *parts: object) -> int:
    key = ":".join([str(seed), *[str(part) for part in parts]]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**31)

