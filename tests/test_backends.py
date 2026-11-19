"""Inference-contract tests with fake models: no weights, network or GPU."""

import contextlib
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "backends.py"
spec = importlib.util.spec_from_file_location("h3_twovoice_test_backends", MODULE_PATH)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.value, axis))

    def detach(self):
        return self

    def float(self):
        return Tensor(self.value.astype(np.float32))

    def cpu(self):
        return self

    def numpy(self):
        return self.value


@pytest.fixture
def fake_runtime(monkeypatch):
    entered = []

    @contextlib.contextmanager
    def inference_mode():
        entered.append(True)
        yield

    torch = types.SimpleNamespace(
        from_numpy=Tensor,
        device=str,
        inference_mode=inference_mode,
        cuda=types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
    )
    monkeypatch.setattr(backend, "_torch", lambda: torch)
    monkeypatch.delenv("H3_TWO_VOICE_DIARIZATION_MODEL", raising=False)
    monkeypatch.delenv("H3_TWO_VOICE_SEPARATOR_MODEL", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return entered


def install_pipeline(monkeypatch, intervals=None, load_error=None, inference_error=None):
    intervals = intervals if intervals is not None else [(0.1, 0.7, "B"), (0.5, 0.9, "A")]
    state = types.SimpleNamespace(devices=[], calls=[], loads=[])

    class Annotation:
        def itertracks(self, yield_label):
            assert yield_label is True
            return [(types.SimpleNamespace(start=a, end=b), i, label) for i, (a, b, label) in enumerate(intervals)]

    class Model:
        def to(self, device):
            state.devices.append(device)

        def __call__(self, audio, **kwargs):
            state.calls.append((audio, kwargs))
            if inference_error:
                raise RuntimeError(inference_error)
            return types.SimpleNamespace(
                speaker_diarization=Annotation(),
                exclusive_speaker_diarization=None,
            )

    class Pipeline:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            state.loads.append((args, kwargs))
            if load_error:
                raise RuntimeError(load_error)
            return Model()

    monkeypatch.setitem(sys.modules, "pyannote", types.ModuleType("pyannote"))
    module = types.ModuleType("pyannote.audio")
    module.Pipeline = Pipeline
    monkeypatch.setitem(sys.modules, "pyannote.audio", module)
    return state


def install_separator(monkeypatch, result_factory=None):
    state = types.SimpleNamespace(devices=[], calls=[], loads=[])

    class Model:
        def to(self, device):
            state.devices.append(device)

        def separate_batch(self, audio):
            state.calls.append(audio.value.copy())
            result = result_factory(audio.value) if result_factory else np.stack([audio.value * 0.25, audio.value * 0.75], axis=-1)
            return Tensor(result)

    class SepformerSeparation:
        @staticmethod
        def from_hparams(**kwargs):
            state.loads.append(kwargs)
            return Model()

    for name in ["speechbrain", "speechbrain.inference"]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    module = types.ModuleType("speechbrain.inference.separation")
    module.SepformerSeparation = SepformerSeparation
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return state


def test_import_does_not_load_inference_packages():
    import ast
    tree = ast.parse(MODULE_PATH.read_text())
    modules = []
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            modules.extend(alias.name for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom):
            modules.append(statement.module)
    assert not any(name.startswith(("torch", "pyannote", "speechbrain", "scipy")) for name in modules)


def test_regular_diarization_keeps_overlap_and_requests_two_speakers(monkeypatch, fake_runtime, tmp_path):
    state = install_pipeline(monkeypatch)
    turns = backend.diarize_turns(np.ones(44101, np.float32) * 0.25, 44100, "cpu", str(tmp_path))
    assert turns == [{"speaker": "B", "start": 0.1, "end": 0.7}, {"speaker": "A", "start": 0.5, "end": 0.9}]
    assert len(fake_runtime) == 1
    audio, kwargs = state.calls[0]
    assert kwargs == {"num_speakers": 2}
    assert audio["sample_rate"] == 16000
    assert audio["waveform"].value.shape == (1, 16000)
    assert audio["waveform"].value.dtype == np.float32
    assert state.devices == ["cpu", "cpu"]


@pytest.mark.parametrize("intervals", [[(0.1, 0.7, "A")], [], [(0.1, 0.2, "A"), (0.3, 0.4, "B"), (0.5, 0.6, "C")]])
def test_requires_exactly_two_speakers(monkeypatch, fake_runtime, tmp_path, intervals):
    state = install_pipeline(monkeypatch, intervals)
    with pytest.raises(RuntimeError, match="exactamente dos"):
        backend.diarize_turns(np.zeros(16000), 16000, "cpu", str(tmp_path))
    assert state.devices[-1] == "cpu"


def test_tokens_and_backend_error_details_are_not_exposed(monkeypatch, fake_runtime, tmp_path):
    secret = "test-private-token-do-not-display"
    monkeypatch.setenv("HF_TOKEN", secret)
    state = install_pipeline(monkeypatch, load_error=secret)
    with pytest.raises(RuntimeError) as error:
        backend.diarize_turns(np.zeros(16000), 16000, "cpu", str(tmp_path))
    assert secret not in str(error.value)
    assert error.value.__suppress_context__
    assert "CC-BY-4.0" in str(error.value)
    assert state.loads[0][1]["token"] == secret


def test_cleanup_after_inference_error(monkeypatch, fake_runtime, tmp_path):
    state = install_pipeline(monkeypatch, inference_error="private detail")
    with pytest.raises(RuntimeError, match="diarización regular"):
        backend.diarize_turns(np.zeros(16000), 16000, "cpu", str(tmp_path))
    assert state.devices == ["cpu", "cpu"]


def test_local_override_and_separate_caches(monkeypatch, fake_runtime, tmp_path):
    local = tmp_path / "diarizer"
    local.mkdir()
    (local / "config.yaml").write_text("{}")
    monkeypatch.setenv("H3_TWO_VOICE_DIARIZATION_MODEL", str(local))
    state = install_pipeline(monkeypatch)
    backend.diarize_turns(np.zeros(16000), 16000, "cpu", str(tmp_path))
    assert state.loads[0][0][0] == str(local)
    assert state.loads[0][1]["token"] is None
    assert backend._cache(str(tmp_path), "x", "one") != backend._cache(str(tmp_path), "x", "two")


def test_missing_local_override_is_not_treated_as_remote(monkeypatch):
    monkeypatch.setenv("H3_TWO_VOICE_SEPARATOR_MODEL", "/does-not-exist/private-model")
    with pytest.raises(RuntimeError, match="modelo local completo") as error:
        backend._source("H3_TWO_VOICE_SEPARATOR_MODEL", backend.SEPARATOR_MODEL, "hyperparams.yaml")
    assert "/does-not-exist" not in str(error.value)


@pytest.mark.parametrize("sample_rate,samples", [(16000, 16003), (44100, 44101), (48000, 240001)])
def test_separator_complete_clip_exact_length_without_peak_normalization(monkeypatch, fake_runtime, tmp_path, sample_rate, samples):
    state = install_separator(monkeypatch)
    audio = np.ones(samples, np.float32) * 0.4
    stems = backend.separate_two(audio, sample_rate, "cpu", str(tmp_path))
    assert len(stems) == 2
    assert all(stem.shape == audio.shape and stem.dtype == np.float32 for stem in stems)
    assert len(state.calls) == 1
    assert state.calls[0].shape == (1, round(samples * 16000 / sample_rate))
    assert len(fake_runtime) == 1
    assert np.mean(stems[0][100:-100]) == pytest.approx(0.1, abs=0.001)
    assert np.mean(stems[1][100:-100]) == pytest.approx(0.3, abs=0.001)
    assert np.max(np.abs(stems[1])) < 0.5
    assert state.loads[0]["source"] == "speechbrain/sepformer-whamr16k"
    assert state.loads[0]["run_opts"] == {"device": "cpu"}
    assert state.devices == ["cpu"]


def test_separator_rejects_long_clip_before_model_load(monkeypatch, fake_runtime, tmp_path):
    state = install_separator(monkeypatch)
    with pytest.raises(ValueError, match="15,5"):
        backend.separate_two(np.zeros(248001), 16000, "cpu", str(tmp_path))
    assert not state.loads


@pytest.mark.parametrize("factory", [lambda x: np.zeros((1, x.shape[1], 3)), lambda x: np.full((1, x.shape[1], 2), np.nan)])
def test_separator_invalid_output_releases_model(monkeypatch, fake_runtime, tmp_path, factory):
    state = install_separator(monkeypatch, factory)
    with pytest.raises(RuntimeError):
        backend.separate_two(np.zeros(16000), 16000, "cpu", str(tmp_path))
    assert state.devices == ["cpu"]


@pytest.mark.parametrize("audio,sr", [(np.zeros((2, 100)), 16000), (np.array([]), 16000), (np.array([np.nan]), 16000), (np.zeros(100), True), (np.zeros(100), 0)])
def test_bad_inputs_fail_before_inference(audio, sr):
    with pytest.raises(ValueError):
        backend.separate_two(audio, sr, "cpu", ".")
