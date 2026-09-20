"""`preload_models` — the option was read but never used.

The trap this guards: `ModelRegistry.pull` reports failure by *yielding* a
final `PullProgress(status="error", …)` rather than raising, so the obvious
`async for _ in registry.pull(name): pass` drains the error and logs
nothing.
"""

import asyncio
from collections.abc import AsyncIterator

import pytest

from litert_server.__main__ import build_app, preload_models
from litert_server.domain.model_registry import PullProgress
from tests.fakes.fake_engine import FakeEngine
from tests.fakes.fake_registry import FakeRegistry


class _Registry(FakeRegistry):
    """Registry whose `pull` ends in whatever progress the test wants."""

    def __init__(self, last: PullProgress | None, *, boom: Exception | None = None) -> None:
        super().__init__()
        self._last = last
        self._boom = boom
        self.pulled: list[str] = []

    async def pull(self, name: str) -> AsyncIterator[PullProgress]:
        self.pulled.append(name)
        if self._boom is not None:
            raise self._boom
        if self._last is not None:
            yield self._last


async def test_configured_models_are_pulled():
    registry = _Registry(PullProgress(bytes_done=1, bytes_total=1, status="done"))
    await preload_models(registry, ["gemma-4-e2b", "qwen3-0.6b"])
    assert registry.pulled == ["gemma-4-e2b", "qwen3-0.6b"]


async def test_yielded_error_is_logged(caplog: pytest.LogCaptureFixture):
    registry = _Registry(
        PullProgress(bytes_done=0, bytes_total=0, status="error", error="401 unauthorized")
    )
    with caplog.at_level("ERROR"):
        await preload_models(registry, ["gemma-4-e2b"])

    assert "401 unauthorized" in caplog.text, "a failed pull was swallowed"


async def test_empty_progress_is_reported_as_failure(caplog: pytest.LogCaptureFixture):
    registry = _Registry(None)
    with caplog.at_level("ERROR"):
        await preload_models(registry, ["gemma-4-e2b"])

    assert "gemma-4-e2b" in caplog.text


async def test_one_failure_does_not_stop_the_rest(caplog: pytest.LogCaptureFixture):
    registry = _Registry(None, boom=RuntimeError("network down"))
    with caplog.at_level("ERROR"):
        await preload_models(registry, ["a", "b"])

    assert registry.pulled == ["a", "b"]
    assert "network down" in caplog.text


async def test_startup_hook_is_started_by_the_lifespan():
    started = asyncio.Event()

    async def hook() -> None:
        started.set()

    app = build_app(engine=FakeEngine(), registry=FakeRegistry(), on_startup=hook)
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(started.wait(), timeout=1)


async def test_startup_does_not_wait_for_the_hook_to_finish():
    """The factory runs before the socket is bound: awaiting a multi-GB
    download here would keep /healthz unreachable and HA would report a
    failed start."""
    release = asyncio.Event()

    async def slow_hook() -> None:
        await release.wait()

    app = build_app(engine=FakeEngine(), registry=FakeRegistry(), on_startup=slow_hook)
    async with asyncio.timeout(1):  # startup must complete while the hook still runs
        async with app.router.lifespan_context(app):
            release.set()
