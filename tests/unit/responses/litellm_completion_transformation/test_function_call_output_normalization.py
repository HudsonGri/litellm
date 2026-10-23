"""
Tests for normalizing Responses API function_call_output into chat tool messages.

This is important for Gemini/Vertex, which expects tool results to be represented
as tool/function response parts; if the tool output is passed as a list of input_* parts,
we normalize it to text/image blocks or a string.
"""

from copy import deepcopy
from typing import Final

import pytest

from litellm.exceptions import BadRequestError

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


def test_function_call_output_list_input_text_is_converted_to_tool_string_content():
    out = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
        tool_call_output={
            "type": "function_call_output",
            "call_id": "call_1",
            "output": [
                {"type": "input_text", "text": "hello"},
                {"type": "input_text", "text": " world"},
            ],
        }
    )

    assert len(out) == 1
    msg = out[0]
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_1"
    assert msg["content"] == "hello world"


def test_function_call_output_string_passthrough():
    out = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
        tool_call_output={
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"ok":true}',
        }
    )
    assert len(out) == 1
    assert out[0]["content"] == '{"ok":true}'


@pytest.mark.parametrize("kind", ["function_call_output", "custom_tool_call_output"])
@pytest.mark.parametrize("detail", ["low", "high", "original"])
@pytest.mark.parametrize("nested_url", [False, True])
def test_tool_output_preserves_image_detail_and_adjacent_text(
    kind: str, detail: str, nested_url: bool,
) -> None:
    image: Final = {
        "type": "input_image",
        "image_url": {"url": "data:image/png;base64,AAAA", "detail": detail}
        if nested_url else "data:image/png;base64,AAAA",
        **({} if nested_url else {"detail": detail}),
    }
    request: Final = {"type": kind, "call_id": "call_media", "output": [
        {"type": "input_text", "text": "Screenshot follows"}, image,
    ]}
    original: Final = deepcopy(request)
    messages: Final = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(request)
    assert messages[0]["content"] == [
        {"type": "text", "text": "Screenshot follows"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": detail}},
    ]
    assert request == original


def test_computer_screenshot_output_remains_structured_media() -> None:
    messages: Final = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message({
        "type": "computer_call_output", "call_id": "call_screenshot",
        "output": {"type": "computer_screenshot", "image_url": "data:image/png;base64,AAAA"},
    })
    assert messages[0]["content"] == [{
        "type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "auto"},
    }]


@pytest.mark.parametrize("tool_output", [False, True])
@pytest.mark.parametrize("mime_field", ["format", "mime_type", "content_type"])
def test_responses_audio_and_file_mime_survive_normalization(tool_output: bool, mime_field: str) -> None:
    parts: Final = [
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
        {"type": "input_file", "file_data": "data:audio/flac;base64,BBBB", mime_field: "audio/flac"},
    ]
    result: Final = (
        LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message({
            "type": "custom_tool_call_output", "call_id": "call_audio", "output": parts,
        })[0]["content"]
        if tool_output else LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(parts)
    )
    assert result == [
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
        {"type": "file", "file": {"file_data": "data:audio/flac;base64,BBBB", "format": "audio/flac"}},
    ]


def _normalize_media_content(kind: str, parts: list[dict[str, object]]) -> object:
    if kind == "direct":
        return LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(parts)
    return LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message({
        "type": kind, "call_id": "call_audio", "output": parts,
    })[0]["content"]


@pytest.mark.parametrize("kind", ["direct", "function_call_output", "custom_tool_call_output"])
@pytest.mark.parametrize("audio_format", ["wav", "mp3", "flac", "pcm", "ogg", "audio/wav", "audio/flac", "audio/ogg"])
def test_responses_audio_formats_and_metadata_survive_in_each_content_path(kind: str, audio_format: str) -> None:
    audio: Final[dict[str, object]] = {
        "type": "input_audio", "input_audio": {"data": "AAAA", "format": audio_format},
        "cache_control": {"type": "ephemeral"},
    }
    parts: Final = [{"type": "input_text", "text": "Transcribe this recording"}, audio]
    original: Final = deepcopy(parts)
    result: Final = _normalize_media_content(kind, parts)
    assert result == [{"type": "text", "text": "Transcribe this recording"}, audio]
    assert parts == original


@pytest.mark.parametrize("kind", ["direct", "function_call_output", "custom_tool_call_output"])
@pytest.mark.parametrize("audio", [
    {"format": "wav"}, {"data": "AAAA"}, {"data": 42, "format": "wav"},
    {"data": "AAAA", "format": 42}, None,
])
def test_responses_audio_rejects_malformed_payloads_in_each_content_path(kind: str, audio: object) -> None:
    parts: Final[list[dict[str, object]]] = [{"type": "input_audio", "input_audio": audio}]
    original: Final = deepcopy(parts)
    with pytest.raises(BadRequestError, match="Invalid Responses media content") as error:
        _normalize_media_content(kind, parts)
    assert error.value.status_code == 400
    assert parts == original


@pytest.mark.parametrize("kind", ["direct", "function_call_output", "custom_tool_call_output"])
@pytest.mark.parametrize("part", [
    {"type": "input_file", "file_data": {"private_payload": "must-not-appear-in-error"}},
    {"type": "input_file", "file_id": "file-test", "filename": 42},
    {"type": "input_file", "file_id": "file-test", "mime_type": ["audio/flac"]},
    {"type": "input_image", "image_url": {42: "must-not-appear-in-error"}},
])
def test_responses_media_validation_failures_are_client_errors(kind: str, part: dict[str, object]) -> None:
    parts: Final = [{"type": "input_text", "text": "Keep adjacent content"}, part]
    original: Final = deepcopy(parts)
    with pytest.raises(BadRequestError) as error:
        _normalize_media_content(kind, parts)
    assert error.value.status_code == 400
    assert "must-not-appear-in-error" not in str(error.value)
    assert parts == original


@pytest.mark.parametrize("kind", ["function_call_output", "custom_tool_call_output"])
@pytest.mark.parametrize("part", [
    {"type": "file"}, {"type": "file", "file": None},
    {"type": "file", "file": {"file_data": 42}},
])
def test_canonical_tool_file_validation_failures_are_client_errors(kind: str, part: dict[str, object]) -> None:
    parts: Final = [part]
    original: Final = deepcopy(parts)
    with pytest.raises(BadRequestError) as error:
        _normalize_media_content(kind, parts)
    assert error.value.status_code == 400
    assert parts == original


@pytest.mark.parametrize("kind", ["direct", "function_call_output", "custom_tool_call_output"])
@pytest.mark.parametrize("part,expected", [
    ({"type": "input_file", "file_url": "https://example.test/a.pdf"},
     {"type": "file", "file": {"file_id": "https://example.test/a.pdf"}}),
    ({"type": "input_file", "file_id": "file-test", "cache_control": {"type": "ephemeral"}},
     {"type": "file", "file": {"file_id": "file-test"}, "cache_control": {"type": "ephemeral"}}),
    ({"type": "input_file", "file_data": "data:audio/flac;base64,AAAA", "content_type": "audio/flac", "detail": "low"},
     {"type": "file", "file": {"file_data": "data:audio/flac;base64,AAAA", "format": "audio/flac", "detail": "low"}}),
    ({"type": "input_file"}, {"type": "file", "file": {}}),
    ({"type": "input_image"}, {"type": "image_url", "image_url": {"url": "", "detail": "auto"}}),
    ({"type": "input_file", "file_data": "data:application/pdf;base64,provider-validates-bytes"},
     {"type": "file", "file": {"file_data": "data:application/pdf;base64,provider-validates-bytes"}}),
])
def test_media_error_mapping_preserves_existing_normalization(
    kind: str, part: dict[str, object], expected: dict[str, object],
) -> None:
    parts: Final = [{"type": "input_text", "text": "Inspect the attachment"}, part]
    original: Final = deepcopy(parts)
    result: Final = _normalize_media_content(kind, parts)
    assert result == [{"type": "text", "text": "Inspect the attachment"}, expected]
    assert parts == original
