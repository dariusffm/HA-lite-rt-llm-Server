from litert_server.domain.prompting import render_chat_prompt
from litert_server.domain.types import ChatTurn


def test_render_chat_prompt_empty():
    assert render_chat_prompt([]) == "<|assistant|>\n"


def test_render_chat_prompt_single_user_turn():
    messages = [ChatTurn(role="user", content="Hello!")]
    expected = "<|user|>\nHello!\n<|assistant|>\n"
    assert render_chat_prompt(messages) == expected


def test_render_chat_prompt_multi_turn_with_system():
    messages = [
        ChatTurn(role="system", content="You are a helpful assistant."),
        ChatTurn(role="user", content="What is the weather?"),
        ChatTurn(role="assistant", content="It's sunny."),
        ChatTurn(role="user", content="Thanks!"),
    ]
    expected = (
        "<|system|>\nYou are a helpful assistant.\n"
        "<|user|>\nWhat is the weather?\n"
        "<|assistant|>\nIt's sunny.\n"
        "<|user|>\nThanks!\n"
        "<|assistant|>\n"
    )
    assert render_chat_prompt(messages) == expected
