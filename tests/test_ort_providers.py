"""Tests for runtime ONNX Runtime provider selection."""

from __future__ import annotations

import logging

import pytest

from livekit.wakeword import _ort_providers
from livekit.wakeword._ort_providers import get_providers


class TestEnvVarOverride:
    def test_single_provider_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LIVEKIT_WAKEWORD_ORT_PROVIDERS", "CPUExecutionProvider")
        assert get_providers() == ["CPUExecutionProvider"]

    def test_multiple_providers_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "LIVEKIT_WAKEWORD_ORT_PROVIDERS",
            "TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider",
        )
        assert get_providers() == [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    def test_whitespace_is_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "LIVEKIT_WAKEWORD_ORT_PROVIDERS",
            "  CUDAExecutionProvider , CPUExecutionProvider  ",
        )
        assert get_providers() == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_empty_string_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty env var must not override auto-detection."""
        monkeypatch.setenv("LIVEKIT_WAKEWORD_ORT_PROVIDERS", "")
        # Should fall through to auto-detect, which always yields at least CPU
        assert "CPUExecutionProvider" in get_providers()

    def test_trailing_comma_tolerated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LIVEKIT_WAKEWORD_ORT_PROVIDERS", "CPUExecutionProvider,")
        assert get_providers() == ["CPUExecutionProvider"]


class TestAutoDetection:
    def test_cpu_only_installation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Plain onnxruntime wheel: CPU is the only option."""
        monkeypatch.delenv("LIVEKIT_WAKEWORD_ORT_PROVIDERS", raising=False)
        monkeypatch.setattr(
            _ort_providers,
            "_DEFAULT_PREFERENCE",
            ("CUDAExecutionProvider", "CPUExecutionProvider"),
        )
        import onnxruntime as ort

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        assert get_providers() == ["CPUExecutionProvider"]

    def test_gpu_installation_prefers_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """onnxruntime-gpu installed: CUDA first, CPU as fallback."""
        monkeypatch.delenv("LIVEKIT_WAKEWORD_ORT_PROVIDERS", raising=False)
        import onnxruntime as ort

        monkeypatch.setattr(
            ort,
            "get_available_providers",
            lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        assert get_providers() == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_ignores_unlisted_available_providers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Providers outside the default preference (e.g. Azure) are not auto-selected."""
        monkeypatch.delenv("LIVEKIT_WAKEWORD_ORT_PROVIDERS", raising=False)
        import onnxruntime as ort

        monkeypatch.setattr(
            ort,
            "get_available_providers",
            lambda: ["AzureExecutionProvider", "CPUExecutionProvider"],
        )
        result = get_providers()
        assert result == ["CPUExecutionProvider"]
        assert "AzureExecutionProvider" not in result

    def test_falls_back_when_no_preferred_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If neither CUDA nor CPU is available, return whatever ORT offers."""
        monkeypatch.delenv("LIVEKIT_WAKEWORD_ORT_PROVIDERS", raising=False)
        import onnxruntime as ort

        monkeypatch.setattr(
            ort,
            "get_available_providers",
            lambda: ["CoreMLExecutionProvider"],
        )
        assert get_providers() == ["CoreMLExecutionProvider"]


class TestLogging:
    def test_logs_auto_selection(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.delenv("LIVEKIT_WAKEWORD_ORT_PROVIDERS", raising=False)
        with caplog.at_level(logging.INFO, logger="livekit.wakeword._ort_providers"):
            get_providers()
        assert any("auto-selected" in r.message for r in caplog.records)

    def test_logs_env_var_override(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("LIVEKIT_WAKEWORD_ORT_PROVIDERS", "CPUExecutionProvider")
        with caplog.at_level(logging.INFO, logger="livekit.wakeword._ort_providers"):
            get_providers()
        assert any("LIVEKIT_WAKEWORD_ORT_PROVIDERS" in r.message for r in caplog.records)


class TestCallSitesUseHelper:
    """Smoke check: the bundled feature extractor actually threads providers through."""

    def test_mel_frontend_uses_helper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Replacing get_providers() changes what MelSpectrogramFrontend passes to ORT."""
        import onnxruntime as ort

        from livekit.wakeword.models import feature_extractor
        from livekit.wakeword.resources import get_mel_model_path

        captured: dict[str, list[str]] = {}
        real_init = ort.InferenceSession

        def spy(path: str, *, providers: list[str], **kwargs: object) -> object:
            captured["providers"] = providers
            return real_init(path, providers=providers, **kwargs)

        monkeypatch.setattr(ort, "InferenceSession", spy)
        monkeypatch.setenv("LIVEKIT_WAKEWORD_ORT_PROVIDERS", "CPUExecutionProvider")

        feature_extractor.MelSpectrogramFrontend(onnx_path=get_mel_model_path())
        assert captured["providers"] == ["CPUExecutionProvider"]
