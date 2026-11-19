"""Real DSP/ComfyUI list contracts with synthetic audio and no model weights.

Only torch's tensor transport and the two inference backends are replaced.
The real segment parser, resampler, phase vocoder and timeline mixer run here.
"""
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return FakeTensor(self.value.astype(np.float32))

    def numpy(self):
        return self.value

    def contiguous(self):
        return FakeTensor(np.ascontiguousarray(self.value))


@pytest.fixture
def nodes(monkeypatch, tmp_path):
    directory = Path(__file__).resolve().parents[1]
    package_name = "_h3_two_voice_contract_tests"
    package = types.ModuleType(package_name)
    package.__path__ = [str(directory)]
    monkeypatch.setitem(sys.modules, package_name, package)

    def unexpected(*args, **kwargs):
        raise AssertionError("An inference backend was called without an explicit test stub")

    backend = types.ModuleType(package_name + ".backends")
    backend.diarize_turns = unexpected
    backend.separate_two = unexpected
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    torch = types.ModuleType("torch")
    torch.Tensor = FakeTensor
    torch.from_numpy = FakeTensor
    monkeypatch.setitem(sys.modules, "torch", torch)
    folders = types.ModuleType("folder_paths")
    folders.models_dir = str(tmp_path / "models")
    monkeypatch.setitem(sys.modules, "folder_paths", folders)

    for name in ("audio_ops", "nodes"):
        spec = importlib.util.spec_from_file_location(package_name + "." + name, directory / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    return module


def tone(sr, seconds, hz=220, amplitude=0.15):
    return (amplitude * np.sin(2 * np.pi * hz * np.arange(round(sr * seconds)) / sr)).astype(np.float32)


def audio(wave, sr=16000):
    wave = np.asarray(wave)
    if wave.ndim == 1:
        wave = wave[None, :]
    return {"waveform": FakeTensor(wave[None]), "sample_rate": sr}


def unpack(result):
    return result["waveform"].numpy()[0]


def frequency(wave, sr):
    wave = wave[len(wave)//5:4*len(wave)//5]
    power = abs(np.fft.rfft(wave * np.hanning(len(wave))))
    return np.fft.rfftfreq(len(wave), 1 / sr)[power.argmax()]


def two_turns(sr=16000):
    source = np.zeros(sr * 3, np.float32)
    source[:sr] = tone(sr, 1, 180)
    source[2*sr:] = tone(sr, 1, 240)
    return source


def prepare_turns(nodes, **kwargs):
    return nodes.H3TwoVoicePrepare().prepare(
        audio(two_turns()), manual_segments="A,0,1\nB,2,3", **kwargs)


def rebuild(nodes, prepared, *, chunks_a=None, chunks_b=None, actions=(2, 2), residual=False):
    a, b, plan, *_ = prepared
    return nodes.H3TwoVoiceRebuild().rebuild(
        [plan], a if chunks_a is None else chunks_a, b if chunks_b is None else chunks_b,
        [actions[0]], [actions[1]], [residual])


def test_comfy_list_contract_and_manual_prepare_without_model(nodes):
    assert nodes.H3TwoVoicePrepare.OUTPUT_IS_LIST == (True, True, False, False, False, False)
    assert nodes.H3TwoVoiceRebuild.INPUT_IS_LIST is True
    a, b, plan, preview_a, preview_b, report = prepare_turns(nodes)
    assert len(a) == len(b) == 1
    assert unpack(a[0]).shape == (1, 16000)
    assert plan["segments"] == {"A": [(0, 16000)], "B": [(32000, 48000)]}
    np.testing.assert_array_equal(unpack(preview_a)[0][:16000], two_turns()[:16000])
    assert np.count_nonzero(unpack(preview_a)[0][16000:]) == 0
    assert np.count_nonzero(unpack(preview_b)[0][:32000]) == 0
    assert "no género" in report


def test_turn_swap_moves_both_audio_and_timestamps(nodes):
    original = prepare_turns(nodes)
    swapped = prepare_turns(nodes, swap_ab=True)
    assert swapped[2]["segments"]["A"] == original[2]["segments"]["B"]
    assert swapped[2]["segments"]["B"] == original[2]["segments"]["A"]
    np.testing.assert_array_equal(unpack(swapped[0][0]), unpack(original[1][0]))
    np.testing.assert_array_equal(unpack(swapped[3]), unpack(original[4]))


def test_automatic_turns_use_anonymous_labels_and_cache_contract(nodes, monkeypatch):
    calls = []

    def diarize(mono, sr, device, cache):
        calls.append((mono.shape, sr, device, cache))
        return [{"speaker": "later", "start": 2, "end": 3},
                {"speaker": "first", "start": 0, "end": 1}]

    monkeypatch.setattr(nodes.backends, "diarize_turns", diarize)
    prepared = nodes.H3TwoVoicePrepare().prepare(audio(two_turns()))
    assert prepared[2]["segments"] == {"A": [(0, 16000)], "B": [(32000, 48000)]}
    assert calls[0][:3] == ((48000,), 16000, "cpu")
    assert calls[0][3].endswith("models/h3_two_voice")


def test_turn_mode_rejects_overlap_without_silently_dropping_a_speaker(nodes):
    with pytest.raises(ValueError, match="solapadas"):
        nodes.H3TwoVoicePrepare().prepare(audio(two_turns()), manual_segments="A,0,2\nB,1,3")


@pytest.mark.parametrize("swap", [False, True])
def test_manual_overlap_preserves_timestamps_and_swap(nodes, monkeypatch, swap):
    sr = 16000
    a, b = np.zeros(sr*3, np.float32), np.zeros(sr*3, np.float32)
    a[:sr*2] = tone(sr, 2, 180)
    b[sr:] = tone(sr, 2, 240)
    monkeypatch.setattr(nodes.backends, "separate_two", lambda *args: [a, b])
    prepared = nodes.H3TwoVoicePrepare().prepare(
        audio(a+b), mode=6, swap_ab=swap, manual_segments="A,0,2\nB,1,3")
    plan = prepared[2]
    expected = {"A": [(sr, sr*3)], "B": [(0, sr*2)]} if swap else {"A": [(0, sr*2)], "B": [(sr, sr*3)]}
    assert plan["segments"] == expected
    np.testing.assert_array_equal(unpack(prepared[3])[0], b if swap else a)
    result, _ = rebuild(nodes, prepared)
    # Identity conversions go through real rebuild; both sources coexist in overlap.
    np.testing.assert_allclose(unpack(result)[0], a+b, atol=1e-7)


def test_overlap_activity_keeps_separate_gaps_and_boundaries(nodes, monkeypatch):
    sr = 16000
    a, b = np.zeros(sr*4, np.float32), np.zeros(sr*4, np.float32)
    a[3200:12800] = tone(sr, 0.6)
    a[40000:49600] = tone(sr, 0.6)
    b[9600:28800] = tone(sr, 1.2, 300)
    monkeypatch.setattr(nodes.backends, "separate_two", lambda *args: [a, b])
    prepared = nodes.H3TwoVoicePrepare().prepare(audio(a+b), mode=6)
    assert len(prepared[0]) == 2 and len(prepared[1]) == 1
    assert prepared[2]["segments"]["A"][0][1] < 32000 < prepared[2]["segments"]["A"][1][0]
    assert prepared[2]["segments"]["B"][0][0] < prepared[2]["segments"]["A"][0][1]


@pytest.mark.parametrize("actions", [(2, 1), (1, 2)])
def test_independent_voice_actions_keep_other_turn_and_gap_exact(nodes, actions):
    prepared = prepare_turns(nodes)
    result, report = rebuild(nodes, prepared, actions=actions,
        chunks_a=[audio(tone(16000, 1, 330))], chunks_b=[audio(tone(16000, 1, 440))])
    wave = unpack(result)[0]
    assert wave.shape == (48000,)
    assert np.count_nonzero(wave[16000:32000]) == 0
    kept = slice(32000, 48000) if actions[1] == 1 else slice(0, 16000)
    np.testing.assert_array_equal(wave[kept], two_turns()[kept])
    converted = wave[:16000] if actions[0] == 2 else wave[32000:]
    assert abs(frequency(converted, 16000) - (330 if actions[0] == 2 else 440)) < 3
    assert "no garantiza" in report


def test_both_preserved_return_exact_stereo_original_including_ambience(nodes):
    mono = two_turns()
    ambience = tone(16000, 3, 90, 0.01)
    source = np.stack([mono + ambience, mono * 0.7 - ambience])
    prepared = nodes.H3TwoVoicePrepare().prepare(audio(source), manual_segments="A,0,1\nB,2,3")
    result, report = nodes.H3TwoVoiceRebuild().rebuild([prepared[2]], None, None, [1], [1], [False])
    np.testing.assert_array_equal(unpack(result), source)
    assert "incluidos canales y ambiente" in report


def test_conversion_resampling_duration_fit_and_original_turn_positions(nodes):
    prepared = prepare_turns(nodes)
    result, _ = rebuild(nodes, prepared,
        chunks_a=[audio(tone(24000, 1.2, 330), 24000)],
        chunks_b=[audio(tone(22050, 0.8, 440), 22050)])
    wave = unpack(result)[0]
    assert result["sample_rate"] == 16000 and len(wave) == 48000
    assert np.count_nonzero(wave[16000:32000]) == 0
    assert abs(frequency(wave[:16000], 16000) - 330) < 4
    assert abs(frequency(wave[32000:], 16000) - 440) < 4


def test_resampling_uses_nearest_sample_length_not_unconditional_ceiling(nodes):
    wave = tone(8000, 1601/8000)
    result = nodes._resample_to(wave, 8000, 11025)
    assert len(result) == round(1601 * 11025 / 8000)


def test_residual_is_opt_in_and_keeps_non_speech_at_original_time(nodes):
    source = two_turns()
    source[16000:32000] = tone(16000, 1, 90, 0.01)
    prepared = nodes.H3TwoVoicePrepare().prepare(audio(source), manual_segments="A,0,1\nB,2,3")
    without, _ = rebuild(nodes, prepared, residual=False)
    with_rest, _ = rebuild(nodes, prepared, residual=True)
    assert np.count_nonzero(unpack(without)[0][16000:32000]) == 0
    np.testing.assert_array_equal(unpack(with_rest)[0][16000:32000], source[16000:32000])


@pytest.mark.parametrize("kind", ["count", "empty", "nan", "silent", "far_too_long", "bad_rate"])
def test_invalid_converted_chunks_fail_instead_of_inserting_false_silence(nodes, kind):
    prepared = prepare_turns(nodes)
    chunks = {
        "count": [],
        "empty": [audio(np.zeros(0))],
        "nan": [audio(np.full(16000, np.nan))],
        "silent": [audio(np.zeros(16000))],
        "far_too_long": [audio(tone(16000, 3))],
        "bad_rate": [audio(tone(16000, 1), 100)],
    }[kind]
    with pytest.raises(ValueError):
        rebuild(nodes, prepared, chunks_a=chunks)


@pytest.mark.parametrize("malformed", ["missing", "wrong_length", "nan"])
def test_bad_separator_output_is_rejected(nodes, monkeypatch, malformed):
    wave = two_turns()
    tracks = {"missing": [wave], "wrong_length": [wave[:-1], wave],
              "nan": [np.full_like(wave, np.nan), wave]}[malformed]
    monkeypatch.setattr(nodes.backends, "separate_two", lambda *args: tracks)
    with pytest.raises(ValueError, match="pistas"):
        nodes.H3TwoVoicePrepare().prepare(audio(wave), mode=6, manual_segments="A,0,1\nB,2,3")


@pytest.mark.parametrize("manual", ["A,0,1", "A,0,0.1\nB,2,3"])
def test_two_meaningful_speakers_are_required(nodes, manual):
    with pytest.raises(ValueError):
        nodes.H3TwoVoicePrepare().prepare(audio(two_turns()), manual_segments=manual)


def test_bad_list_contract_and_action_fail_clearly(nodes):
    a, b, plan, *_ = prepare_turns(nodes)
    with pytest.raises(ValueError, match="único valor"):
        nodes.H3TwoVoiceRebuild().rebuild([plan, plan], a, b, [2], [2], [False])
    with pytest.raises(ValueError, match="conservar"):
        nodes.H3TwoVoiceRebuild().rebuild([plan], a, b, [3], [2], [False])


@pytest.mark.parametrize("wave", [np.zeros(16000), np.full(16000, np.inf), np.ones(16000*16)])
def test_silent_invalid_or_excessively_long_sources_fail_before_backend(nodes, wave):
    with pytest.raises(ValueError):
        nodes.H3TwoVoicePrepare().prepare(audio(wave))
