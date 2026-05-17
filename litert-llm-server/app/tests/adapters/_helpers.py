"""Shared test helpers for adapter HTTP tests."""

from __future__ import annotations

import json
from typing import Any

from httpx import AsyncClient


async def post_ndjson(
    client: AsyncClient, url: str, payload: dict[str, Any]
) -> list[dict[str, Any]]:
    """POST a JSON payload, assert NDJSON content-type, return decoded records."""
    async with client.stream("POST", url, json=payload) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/x-ndjson")
        lines = [line async for line in r.aiter_lines() if line.strip()]
    return [json.loads(line) for line in lines]
