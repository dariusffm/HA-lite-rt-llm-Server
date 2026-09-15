from collections.abc import AsyncIterator

from litert_server.domain.inference import InferenceService, collect_chat
from litert_server.domain.types import ChatTurn, GenerationParams, Token, ToolCall, ToolSpec


class _Concrete:
    async def stream_completion(
        self,
        model: str,
        prompt: str,
        params: GenerationParams,
    ) -> AsyncIterator[Token]:
        yield Token(text=prompt, index=0, finish_reason="stop")


def test_concrete_is_structural_subtype():
    svc: InferenceService = _Concrete()
    assert svc is not None


class _ToolEngine:
    """Minimal engine that answers with a tool call and records ``tools``."""

    def __init__(self) -> None:
        self.tools: list[ToolSpec] | None = None

    async def stream_completion(self, model, prompt, params):
        yield Token(text="", index=0, finish_reason="stop")

    async def stream_chat(self, model, messages, params, tools=None):
        self.tools = tools
        yield Token(
            text="",
            index=0,
            finish_reason="tool_calls",
            tool_calls=[ToolCall(id="call_1", name="get_weather", arguments={"city": "Frankfurt"})],
        )


async def test_collect_chat_returns_tool_calls():
    engine = _ToolEngine()
    spec = ToolSpec(name="get_weather", parameters={"type": "object", "properties": {}})
    params = GenerationParams(max_tokens=10, temperature=0.1)
    text, finish, calls = await collect_chat(
        engine, "m", [ChatTurn(role="user", content="?")], params, tools=[spec]
    )
    assert text == ""
    assert finish == "tool_calls"
    assert calls is not None and calls[0].name == "get_weather"
    assert engine.tools == [spec]
