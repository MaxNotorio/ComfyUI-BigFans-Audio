"""Signal/timeline tests: no voice models, recordings, or paid generation."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("audio_ops", Path(__file__).resolve().parents[1] / "audio_ops.py")
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


def tone(sr, seconds, frequency=220, amplitude=0.2):
    t = np.arange(round(sr * seconds)) / sr
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def peak_frequency(audio, sr):
    middle = audio[len(audio)//5:4*len(audio)//5]
    spectrum = np.abs(np.fft.rfft(middle * np.hanning(len(middle))))
    return np.fft.rfftfreq(len(middle), 1 / sr)[np.argmax(spectrum)]


def test_manual_sample_rounding_and_explicit_labels():
    records = ops.parse_manual_segments("B,0,0.3333\nA,0.5,1", 24000, 24000)
    assert records == [
        {"speaker": "B", "start": 0, "end": 7999},
        {"speaker": "A", "start": 12000, "end": 24000},
    ]


def test_automatic_labels_follow_first_occurrence_and_merge_same_speaker():
    records = [
        {"speaker": "spk_8", "start": 1200, "end": 2500},
        {"speaker": "spk_2", "start": 0, "end": 500},
        {"speaker": "spk_2", "start": 450, "end": 1000},
    ]
    normalized = ops.normalize_segments(records, 16000, 3000)
    assert normalized == [{"speaker": "A", "start": 0, "end": 1000},
                          {"speaker": "B", "start": 1200, "end": 2500}]
    assert records[0]["speaker"] == "spk_8"  # No mutation of input annotations.


def test_turns_reject_real_overlap_but_overlap_mode_retains_both():
    records = [{"speaker": "a", "start": 0, "end": 12000},
               {"speaker": "b", "start": 8000, "end": 16000}]
    with pytest.raises(ValueError, match="solapadas"):
        ops.normalize_segments(records, 16000, 16000)
    result = ops.normalize_segments(records, 16000, 16000, reject_overlap=False)
    assert result[0]["end"] == 12000 and result[1]["start"] == 8000


def test_tiny_boundary_overlap_splits_without_double_conversion():
    records = [{"speaker": "a", "start": 0, "end": 8000},
               {"speaker": "b", "start": 7800, "end": 16000}]
    result = ops.normalize_segments(records, 16000, 16000)
    assert result[0]["end"] == result[1]["start"] == 7900


@pytest.mark.parametrize("text", ["A,0,2", "C,0,1", "A,NaN,1", "A,0.5,0.1", "A;0;1", "A,-0.1,0.2"])
def test_manual_invalid_bounds_and_syntax(text):
    with pytest.raises(ValueError):
        ops.parse_manual_segments(text, 16000, 16000)


def test_activity_keeps_long_silence_and_merges_short_intra_voice_gap():
    sr = 16000
    audio = np.zeros(sr * 4, dtype=np.float32)
    audio[sr//2:sr] = tone(sr, 0.5)
    audio[round(sr*1.08):round(sr*1.5)] = tone(sr, 0.42)
    audio[round(sr*2.7):round(sr*3.4)] = tone(sr, 0.7)
    regions = ops.activity_segments(audio, sr)
    assert len(regions) == 2
    assert abs(regions[0][0]/sr - 0.46) < 0.02
    assert abs(regions[0][1]/sr - 1.54) < 0.02
    assert regions[0][1] < 2 * sr < regions[1][0]
    assert ops.activity_segments(np.zeros(sr, np.float32), sr) == []


def test_activity_uses_channel_energy_without_antiphase_cancellation():
    wave = tone(16000, 1)
    regions = ops.activity_segments(np.stack([wave, -wave]), 16000)
    assert regions == [(0, 16000)]


@pytest.mark.parametrize("seconds", [0.7, 1.3])
def test_time_fit_changes_length_without_pitch_shift(seconds):
    sr = 16000
    converted = tone(sr, seconds, frequency=330)
    result = ops.fit_clip(converted, sr, sr)
    assert len(result) == sr
    assert abs(peak_frequency(result, sr) - 330) < 4
    assert np.isfinite(result).all()


def test_converted_turns_return_to_their_original_positions():
    sr = 16000
    original = np.zeros(sr * 4, dtype=np.float32)
    intervals = [(sr//2, round(sr*1.2)), (round(sr*2.5), round(sr*3.3))]
    original[intervals[0][0]:intervals[0][1]] = tone(sr, 0.7, 180)
    original[intervals[1][0]:intervals[1][1]] = tone(sr, 0.8, 180)
    result = ops.build_track(original, sr, intervals, [tone(sr, 0.9, 280), tone(sr, 0.6, 280)], 2)
    assert result.shape == original.shape
    assert np.all(result[:sr//2] == 0)
    assert np.all(result[round(sr*1.2):round(sr*2.5)] == 0)
    assert np.all(result[round(sr*3.3):] == 0)
    assert abs(peak_frequency(result[intervals[0][0]:intervals[0][1]], sr) - 280) < 5


def test_two_overlapping_tracks_mix_at_original_offsets():
    sr = 16000
    a, b = np.zeros(sr*3, np.float32), np.zeros(sr*3, np.float32)
    a[:sr*2] = tone(sr, 2, 170)
    b[sr:] = tone(sr, 2, 250)
    new_a = ops.build_track(a, sr, [(0, sr*2)], [tone(sr, 2.3, 300)], 2)
    new_b = ops.build_track(b, sr, [(sr, sr*3)], [tone(sr, 1.7, 440)], 2)
    mixture = new_a + new_b
    assert len(mixture) == sr*3
    assert np.max(np.abs(new_b[:sr])) == 0
    assert np.max(np.abs(new_a[sr*2:])) == 0
    assert np.mean(mixture[sr:sr*2]**2) > 0.01


def test_preserve_is_bit_exact_and_requires_no_conversion():
    original = np.random.default_rng(41).normal(0, 0.1, (2, 18000))
    result = ops.build_track(original, 24000, [], None, 1)
    assert np.array_equal(original, result)
    assert result.dtype == original.dtype
    assert result is not original


def test_mono_conversion_can_replace_stereo_source_and_gain_is_bounded():
    sr = 16000
    original = np.stack([tone(sr, 1), tone(sr, 1)])
    result = ops.build_track(original, sr, [(0, sr)], [tone(sr, 1, amplitude=0.0001)], 2, fade_ms=0)
    assert result.shape == original.shape
    assert np.max(np.abs(result)) < 0.00041


@pytest.mark.parametrize("converted", [np.array([], dtype=np.float32), np.full(16000, np.nan), np.zeros(16000)])
def test_invalid_or_silent_conversion_is_not_silently_mounted(converted):
    with pytest.raises(ValueError):
        ops.fit_clip(converted, 16000, 16000)


def test_invalid_segment_count_overlap_and_duration_drift():
    audio = tone(16000, 1)
    with pytest.raises(ValueError, match="esperaban"):
        ops.build_track(audio, 16000, [(0, 16000)], [], 2)
    with pytest.raises(ValueError, match="solaparse"):
        ops.build_track(audio, 16000, [(0, 10000), (9000, 16000)], [audio, audio], 2)
    with pytest.raises(ValueError, match="demasiado"):
        ops.fit_clip(tone(16000, 3), 16000, 16000)


def test_resampling_restores_exact_length_and_pitch():
    source = tone(16000, 1.125, 300)
    result = ops.resample_audio(source, 16000, 48000)
    assert len(result) == 54000
    assert abs(peak_frequency(result, 48000) - 300) < 4
    assert np.array_equal(ops.resample_audio(source, 16000, 16000), source)


@pytest.mark.parametrize("sr", [0, -1, 16000.5, True])
def test_invalid_sample_rate(sr):
    with pytest.raises(ValueError):
        ops.activity_segments(np.ones(16000), sr)
