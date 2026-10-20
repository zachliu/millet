"""Tests for the opt-in AWS Bedrock summary backend.

Covers what the backend must get right:
  * chain wiring -- bedrock is selectable but never in the default chain;
  * request shape -- system prompt, image frames, effort, no temperature;
  * result handling -- refusal / truncation / empty fail loud, refusal
    fallback is recorded in provenance;
  * PDF -- a Bedrock summary never gets the TEE attestation footer.

No network, no AWS: the Mantle client is replaced with a fake.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

import millet.summarize as sm


class _FakeStream:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._response


def _response(text="## Summary\nok", stop_reason="end_turn", iterations=None):
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        stop_reason=stop_reason,
        usage=SimpleNamespace(iterations=iterations),
    )


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Install a fake ``anthropic`` module; returns the captured calls."""
    calls: dict = {"response": _response()}

    class _Messages:
        def stream(self, **kwargs):
            calls["stream_kwargs"] = kwargs
            return _FakeStream(calls["response"])

    class _Client:
        def __init__(self, **kwargs):
            calls["client_kwargs"] = kwargs
            self.beta = SimpleNamespace(messages=_Messages())

    class _Middleware:
        def __init__(self, fallbacks):
            calls["fallbacks"] = list(fallbacks)

    mod = types.ModuleType("anthropic")
    mod.AnthropicBedrockMantle = _Client
    mod.BetaRefusalFallbackMiddleware = _Middleware
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.delenv("MILLET_BEDROCK_REGION", raising=False)
    monkeypatch.delenv("MILLET_BEDROCK_EFFORT", raising=False)
    return calls


def _cfg(**kw):
    return sm.SummaryConfig(backend="bedrock", **kw)


class TestRegistry:
    def test_selectable_but_not_in_default_chain(self):
        assert "bedrock" in sm.BACKENDS
        assert "bedrock" not in sm.DEFAULT_FALLBACK_ORDER

    def test_default_model(self):
        assert sm._default_model_for_backend("bedrock") == "anthropic.claude-opus-5-5"
        assert _cfg().model == "anthropic.claude-opus-5-5"

    def test_vision_capable(self):
        assert sm.backend_supports_vision("bedrock") is True

    def test_ollama_primary_never_falls_back_to_bedrock(self, monkeypatch):
        """Even with AWS credentials present, bedrock is not tried unless chosen."""
        tried: list[str] = []
        monkeypatch.setattr(sm, "is_backend_available", lambda cfg: True)

        def _dispatch(backend, *a, **k):
            tried.append(backend)
            raise RuntimeError(f"{backend} boom")

        monkeypatch.setattr(sm, "_dispatch", _dispatch)
        monkeypatch.delenv("MILLET_SUMMARY_FALLBACK_ORDER", raising=False)
        with pytest.raises(RuntimeError):
            sm.summarize("hello", sm.SummaryConfig(backend="ollama"))
        assert "bedrock" not in tried

    def test_explicit_fallback_order_can_include_bedrock(self, monkeypatch):
        monkeypatch.setenv("MILLET_SUMMARY_FALLBACK_ORDER", "bedrock,ollama")
        assert sm._resolve_fallback_order() == ("bedrock", "ollama")


class TestAvailability:
    def test_bearer_token_counts_as_credentials(self, monkeypatch):
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "tok")
        assert sm._bedrock_credentials_available() is True

    def test_missing_sdk_message(self, monkeypatch):
        monkeypatch.setattr(sm, "bedrock_sdk_installed", lambda: False)
        cfg = _cfg()
        assert sm.is_backend_available(cfg) is False
        assert "millet-pipeline[bedrock]" in sm._backend_not_available_message(cfg)


class TestRequest:
    def test_request_shape(self, fake_anthropic):
        result = sm._summarize_bedrock("SYS", "USER", _cfg())
        kw = fake_anthropic["stream_kwargs"]
        assert kw["model"] == "anthropic.claude-opus-5-5"
        assert kw["system"] == "SYS"
        assert kw["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "USER"}]}
        ]
        assert kw["output_config"] == {"effort": "medium"}
        assert "temperature" not in kw
        assert fake_anthropic["fallbacks"] == [{"model": sm.BEDROCK_REFUSAL_FALLBACK_MODEL}]
        assert result.backend == "bedrock"
        assert result.model == "anthropic.claude-opus-5-5 (Bedrock)"
        assert result.markdown == "## Summary\nok"
        assert result.fallback_used is False
        assert result.frames_used == 0

    def test_region_and_effort_overrides(self, fake_anthropic, monkeypatch):
        monkeypatch.setenv("MILLET_BEDROCK_REGION", "eu-west-1")
        monkeypatch.setenv("MILLET_BEDROCK_EFFORT", "")
        sm._summarize_bedrock("SYS", "USER", _cfg())
        assert fake_anthropic["client_kwargs"]["aws_region"] == "eu-west-1"
        assert "output_config" not in fake_anthropic["stream_kwargs"]

    def test_frames_sent_as_image_blocks(self, fake_anthropic, tmp_path):
        png = tmp_path / "a.png"
        png.write_bytes(b"\x89PNG\r\n")
        jpg = tmp_path / "b.jpg"
        jpg.write_bytes(b"\xff\xd8\xff")
        result = sm._summarize_bedrock("SYS", "USER", _cfg(frames=[png, jpg]))
        content = fake_anthropic["stream_kwargs"]["messages"][0]["content"]
        assert [c["type"] for c in content] == ["image", "image", "text"]
        assert content[0]["source"]["media_type"] == "image/png"
        assert content[1]["source"]["media_type"] == "image/jpeg"
        assert result.frames_used == 2


class TestResultHandling:
    def test_refusal_raises(self, fake_anthropic):
        fake_anthropic["response"] = _response(stop_reason="refusal")
        with pytest.raises(RuntimeError, match="declined"):
            sm._summarize_bedrock("SYS", "USER", _cfg())

    def test_truncation_raises(self, fake_anthropic):
        fake_anthropic["response"] = _response(stop_reason="max_tokens")
        with pytest.raises(RuntimeError, match="max_tokens"):
            sm._summarize_bedrock("SYS", "USER", _cfg())

    def test_empty_raises(self, fake_anthropic):
        fake_anthropic["response"] = _response(text="  ")
        with pytest.raises(RuntimeError, match="empty"):
            sm._summarize_bedrock("SYS", "USER", _cfg())

    def test_refusal_fallback_recorded(self, fake_anthropic):
        fake_anthropic["response"] = _response(
            iterations=[SimpleNamespace(type="message"), SimpleNamespace(type="fallback_message")]
        )
        result = sm._summarize_bedrock("SYS", "USER", _cfg())
        assert result.fallback_used is True
        assert result.model == f"{sm.BEDROCK_REFUSAL_FALLBACK_MODEL} (Bedrock)"

    def test_summarize_end_to_end(self, fake_anthropic, monkeypatch):
        monkeypatch.setattr(sm, "is_backend_available", lambda cfg: cfg.backend == "bedrock")
        result = sm.summarize("YOU: hello", _cfg())
        assert result.backend == "bedrock"
        assert result.fallback_used is False


def test_pdf_not_attested_for_bedrock(monkeypatch, tmp_path, transcript):
    import millet.pdf as pdf

    seen: dict = {}
    real = pdf._PDFDocTemplate

    class _Spy(real):
        def __init__(self, *a, **kw):
            seen["attested"] = kw.get("attested")
            super().__init__(*a, **kw)

    monkeypatch.setattr(pdf, "_PDFDocTemplate", _Spy)
    summary = sm.MeetingSummary(
        markdown="## Summary\nok", model="m (Bedrock)", elapsed_seconds=1.0, backend="bedrock"
    )
    pdf.generate_pdf(transcript, tmp_path / "out.pdf", summary=summary)
    assert seen["attested"] is False


class TestRegion:
    _VARS = ("MILLET_BEDROCK_REGION", "AWS_REGION", "AWS_DEFAULT_REGION",
             "ANTHROPIC_BEDROCK_MANTLE_BASE_URL")

    def _clear(self, monkeypatch):
        for v in self._VARS:
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("AWS_CONFIG_FILE", "/nonexistent")  # no profile region

    def test_defaults_to_us_east_1(self, monkeypatch):
        self._clear(monkeypatch)
        assert sm._resolve_bedrock_region() == "us-east-1"

    def test_precedence(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
        assert sm._resolve_bedrock_region() == "us-west-2"
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        assert sm._resolve_bedrock_region() == "eu-west-1"
        monkeypatch.setenv("MILLET_BEDROCK_REGION", "ap-northeast-1")
        assert sm._resolve_bedrock_region() == "ap-northeast-1"

    def test_base_url_leaves_region_to_sdk(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_BEDROCK_MANTLE_BASE_URL", "https://example.test")
        assert sm._resolve_bedrock_region() is None

    def test_client_gets_default_region(self, fake_anthropic, monkeypatch):
        self._clear(monkeypatch)
        sm._summarize_bedrock("SYS", "USER", _cfg())
        assert fake_anthropic["client_kwargs"]["aws_region"] == "us-east-1"
