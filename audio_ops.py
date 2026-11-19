"""Timing utilities for two-speaker voice conversion.

All intervals use half-open sample coordinates [start, end). Arrays use time on
the last axis: [samples] or [channels, samples]. This module does not load models,
identify gender, or infer which visible character owns a voice.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from numbers import Integral
import math

import numpy as np


def _sample_rate(sr: int) -> int:
    if isinstance(sr, bool) or not isinstance(sr, Integral) or not 1000 <= sr <= 384000:
        raise ValueError("sample_rate debe ser un entero entre 1000 y 384000 Hz.")
    return int(sr)


def _audio(value, name="audio") -> np.ndarray:
    array = np.asarray(value)
    if array.ndim not in (1, 2) or array.shape[-1] == 0:
        raise ValueError(f"{name}: se espera audio no vacío [muestras] o [canales, muestras].")
    if array.ndim == 2 and not 1 <= array.shape[0] <= 8:
        raise ValueError(f"{name}: la primera dimensión debe contener de 1 a 8 canales.")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{name}: las muestras deben ser números reales.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: contiene NaN o infinito.")
    return array


def _index(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name}: se requieren índices de muestra enteros.")
    return int(value)


def _interval(record, n_samples: int) -> tuple[int, int]:
    if isinstance(record, Mapping):
        start, end = record["start"], record["end"]
    else:
        if len(record) != 2:
            raise ValueError("Cada intervalo debe contener inicio y final.")
        start, end = record
    start, end = _index(start, "start"), _index(end, "end")
    if not 0 <= start < end <= n_samples:
        raise ValueError(f"Intervalo [{start}, {end}) fuera de los {n_samples} samples o vacío.")
    return start, end


def normalize_segments(
    records: Iterable[Mapping], sr: int, n_samples: int, *,
    reject_overlap: bool = True, overlap_tolerance_ms: float = 40.0,
    relabel: bool = True,
) -> list[dict]:
    """Validate exact sample intervals and canonicalize at most two speakers.

    With relabel=True, the first distinct speaker heard becomes A, then B.
    Same-speaker overlapping/touching intervals merge. In turn-taking mode a
    cross-speaker overlap beyond tolerance raises rather than silently dropping
    words. Tiny boundary overlaps are split at their midpoint, so they cannot be
    converted twice. With reject_overlap=False, cross-speaker overlap remains.
    """
    sr = _sample_rate(sr)
    n_samples = _index(n_samples, "n_samples")
    if n_samples <= 0:
        raise ValueError("El audio no contiene muestras.")
    if not math.isfinite(overlap_tolerance_ms) or overlap_tolerance_ms < 0:
        raise ValueError("overlap_tolerance_ms debe ser finito y no negativo.")
    items = []
    for record in records:
        if not isinstance(record, Mapping) or not {"start", "end", "speaker"} <= record.keys():
            raise ValueError("Cada segmento requiere speaker, start y end.")
        start, end = _interval(record, n_samples)
        label = str(record["speaker"]).strip()
        if not label:
            raise ValueError("Un segmento no tiene etiqueta de hablante.")
        items.append({"speaker": label, "start": start, "end": end})
    if not items:
        raise ValueError("No se detectaron intervenciones; revisa el audio o escribe los intervalos manuales.")
    items.sort(key=lambda r: (r["start"], r["end"], r["speaker"]))
    labels = list(dict.fromkeys(r["speaker"] for r in items))
    if len(labels) > 2:
        raise ValueError(f"Se detectaron {len(labels)} hablantes; esta versión admite un máximo de dos.")
    if relabel:
        mapping = {label: "AB"[i] for i, label in enumerate(labels)}
    else:
        if any(label not in ("A", "B") for label in labels):
            raise ValueError("Las etiquetas manuales deben ser A o B.")
        mapping = {label: label for label in labels}
    for record in items:
        record["speaker"] = mapping[record["speaker"]]

    merged = []
    for speaker in ("A", "B"):
        speaker_items = [r for r in items if r["speaker"] == speaker]
        for record in speaker_items:
            if merged and merged[-1]["speaker"] == speaker and record["start"] <= merged[-1]["end"]:
                merged[-1]["end"] = max(merged[-1]["end"], record["end"])
            else:
                merged.append(dict(record))
    merged.sort(key=lambda r: (r["start"], r["end"], r["speaker"]))

    if reject_overlap:
        tolerance = int(round(overlap_tolerance_ms * sr / 1000.0))
        for i, left in enumerate(merged):
            for right in merged[i + 1:]:
                if right["start"] >= left["end"]:
                    break
                if left["speaker"] == right["speaker"]:
                    continue
                overlap = min(left["end"], right["end"]) - right["start"]
                # A nested intervention cannot be repaired with one boundary.
                nested = right["end"] <= left["end"]
                if overlap > tolerance or nested:
                    raise ValueError(
                        f"Voces solapadas entre {right['start']/sr:.3f} y "
                        f"{min(left['end'], right['end'])/sr:.3f} s. "
                        "Usa el modo de voces solapadas o corrige los intervalos."
                    )
                boundary = (left["end"] + right["start"]) // 2
                left["end"] = boundary
                right["start"] = boundary
    return merged


def _seconds_to_samples(value, sr: int) -> int:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"Tiempo no válido: {value!r}") from None
    if not number.is_finite():
        raise ValueError("Los tiempos deben ser finitos.")
    return int((number * sr).to_integral_value(rounding=ROUND_HALF_UP))


def segments_from_seconds(records, sr: int, n_samples: int, **kwargs) -> list[dict]:
    """Convert model annotations in seconds into exact sample intervals."""
    sr = _sample_rate(sr)
    converted = [
        {"speaker": r["speaker"], "start": _seconds_to_samples(r["start"], sr),
         "end": _seconds_to_samples(r["end"], sr)}
        for r in records
    ]
    return normalize_segments(converted, sr, n_samples, **kwargs)


def parse_manual_segments(text: str, sr: int, n_samples: int, **kwargs) -> list[dict]:
    """Parse one strict 'A,0,2.4' or 'B,2.6,5' record per nonempty line.

    Explicit manual A/B labels remain fixed even when B speaks first.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("No hay intervalos manuales.")
    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3 or parts[0] not in ("A", "B"):
            raise ValueError(f"Línea {line_number}: usa A,inicio,final o B,inicio,final, con segundos decimales.")
        records.append({"speaker": parts[0], "start": parts[1], "end": parts[2]})
    kwargs.setdefault("relabel", False)
    return segments_from_seconds(records, sr, n_samples, **kwargs)


def activity_segments(
    track, sr: int, *, threshold_db: float = -40.0, frame_ms: float = 20.0,
    hop_ms: float = 10.0, merge_gap_ms: float = 160.0,
    padding_ms: float = 30.0, min_segment_ms: float = 180.0,
) -> list[tuple[int, int]]:
    """Estimate active regions on ONE already separated voice using RMS energy.

    This is not speaker recognition or a trained VAD. Empty/silent input returns
    no regions so the caller can report it, instead of inventing speech. Channel
    energies are averaged (waveforms are not summed, avoiding phase cancellation).
    """
    sr = _sample_rate(sr)
    source = _audio(track, "pista separada")
    for name, val in {"threshold_db": threshold_db, "frame_ms": frame_ms,
                      "hop_ms": hop_ms, "merge_gap_ms": merge_gap_ms,
                      "padding_ms": padding_ms, "min_segment_ms": min_segment_ms}.items():
        if not math.isfinite(val):
            raise ValueError(f"{name} debe ser finito.")
    if threshold_db >= 0 or frame_ms <= 0 or hop_ms <= 0 or min(merge_gap_ms, padding_ms, min_segment_ms) < 0:
        raise ValueError("Umbral/ventanas de actividad no válidos.")
    n = source.shape[-1]
    frame = max(1, round(sr * frame_ms / 1000))
    hop = max(1, round(sr * hop_ms / 1000))
    energy = np.square(source.astype(np.float64))
    if energy.ndim == 2:
        energy = energy.mean(axis=0)
    cumulative = np.concatenate(([0.0], np.cumsum(energy)))
    starts = np.arange(0, n, hop, dtype=np.int64)
    stops = np.minimum(starts + frame, n)
    rms = np.sqrt(np.maximum(0.0, (cumulative[stops] - cumulative[starts]) / (stops - starts)))
    peak = float(rms.max())
    if peak < 1e-7:
        return []
    active = rms >= max(1e-7, peak * 10 ** (threshold_db / 20))
    intervals = []
    for start, stop in zip(starts[active].tolist(), stops[active].tolist()):
        if intervals and start <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], stop)
        else:
            intervals.append([start, stop])
    gap = round(sr * merge_gap_ms / 1000)
    merged = []
    for start, stop in intervals:
        if merged and start - merged[-1][1] <= gap:
            merged[-1][1] = stop
        else:
            merged.append([start, stop])
    minimum = max(1, round(sr * min_segment_ms / 1000))
    padding = round(sr * padding_ms / 1000)
    padded = []
    for start, stop in merged:
        if stop - start < minimum:
            continue
        start, stop = max(0, start - padding), min(n, stop + padding)
        if padded and start <= padded[-1][1]:
            padded[-1] = (padded[-1][0], max(padded[-1][1], stop))
        else:
            padded.append((start, stop))
    return padded


def resample_audio(audio, source_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resampling, with deterministic length and no pitch remapping."""
    source_sr, target_sr = _sample_rate(source_sr), _sample_rate(target_sr)
    source = _audio(audio)
    if source_sr == target_sr:
        return source.copy()
    from scipy.signal import resample_poly
    divisor = math.gcd(source_sr, target_sr)
    result = resample_poly(source.astype(np.float32), target_sr // divisor,
                           source_sr // divisor, axis=-1)
    target_length = max(1, int(round(source.shape[-1] * target_sr / source_sr)))
    return _fix_length(result, target_length)


def _fix_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    if audio.shape[-1] >= target_length:
        return audio[..., :target_length].copy()
    padding = [(0, 0)] * audio.ndim
    padding[-1] = (0, target_length - audio.shape[-1])
    return np.pad(audio, padding)


def fit_clip(
    clip, target_samples: int, sr: int, *, mode: str = "stretch",
    min_ratio: float = 0.5, max_ratio: float = 2.0,
) -> np.ndarray:
    """Fit a converted utterance to its own original interval.

    Stretch uses a phase vocoder instead of changing the sample rate, preserving
    nominal pitch. Exact duration does not establish phoneme-level lip sync.
    trim_pad is an explicit alternative and may cut speech; it is never chosen
    silently. Large timing changes raise for review.
    """
    _sample_rate(sr)
    source = _audio(clip, "fragmento convertido").astype(np.float32)
    target_samples = _index(target_samples, "target_samples")
    if target_samples <= 0:
        raise ValueError("La longitud de destino debe ser positiva.")
    if mode not in ("stretch", "trim_pad"):
        raise ValueError("fit debe ser stretch o trim_pad.")
    ratio = source.shape[-1] / target_samples
    if not (math.isfinite(min_ratio) and math.isfinite(max_ratio) and 0 < min_ratio <= max_ratio):
        raise ValueError("Los límites de ajuste temporal no son válidos.")
    if not min_ratio <= ratio <= max_ratio:
        raise ValueError(f"La conversión cambió demasiado la duración (factor {ratio:.2f}); revisa el audio antes de sincronizar.")
    if float(np.max(np.abs(source))) < 1e-7:
        raise ValueError("El conversor devolvió silencio; no se sustituirá una voz por una salida vacía.")
    if source.shape[-1] == target_samples:
        return source.copy()
    if mode == "trim_pad":
        return _fix_length(source, target_samples)
    if min(source.shape[-1], target_samples) < max(32, int(sr * 0.04)):
        raise ValueError("Fragmento demasiado corto para ajustar su duración sin cambiar el tono.")
    from librosa.effects import time_stretch
    n_fft = min(2048, 2 ** int(math.floor(math.log2(source.shape[-1]))))
    result = time_stretch(source, rate=ratio, n_fft=n_fft, hop_length=max(8, n_fft // 4))
    return _fix_length(_audio(result, "ajuste temporal"), target_samples).astype(np.float32)


def _match_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    source = audio[np.newaxis, :] if audio.ndim == 1 else audio
    if source.shape[0] == channels:
        return source
    if channels == 1:
        return source.mean(axis=0, keepdims=True)
    if source.shape[0] == 1:
        return np.repeat(source, channels, axis=0)
    raise ValueError("El número de canales convertido no coincide con el destino.")


def build_track(
    original_track, sr: int, segments, converted_clips, action: int, *,
    fit: str = "stretch", fade_ms: float = 8.0, rms_match: bool = True,
    max_gain_db: float = 12.0,
) -> np.ndarray:
    """Rebuild one speaker on the original clock, never concatenate utterances.

    action=1 returns an exact copy without requiring converted clips. action=2
    replaces each selected interval and leaves all samples outside it intact.
    Supply a masked/separated track, not the full conversation, for per-speaker
    mixing. converted_clips must already have the same sample rate as sr.
    """
    sr = _sample_rate(sr)
    original = _audio(original_track, "pista original")
    if action == 1:
        return original.copy()
    if action != 2:
        raise ValueError("La acción debe ser 1 (conservar) o 2 (convertir).")
    intervals = [_interval(s, original.shape[-1]) for s in segments]
    if not intervals:
        raise ValueError("No hay intervenciones de esta voz para convertir.")
    if intervals != sorted(intervals):
        raise ValueError("Los fragmentos deben estar ordenados por su inicio.")
    if any(b[0] < a[1] for a, b in zip(intervals, intervals[1:])):
        raise ValueError("Los fragmentos de una misma voz no pueden solaparse.")
    clips = list(converted_clips)
    if len(clips) != len(intervals):
        raise ValueError(f"Se esperaban {len(intervals)} fragmentos convertidos, llegaron {len(clips)}.")
    if not math.isfinite(fade_ms) or not 0 <= fade_ms <= 8:
        raise ValueError("fade_ms debe estar entre 0 y 8 ms.")
    if not math.isfinite(max_gain_db) or not 0 <= max_gain_db <= 24:
        raise ValueError("max_gain_db debe estar entre 0 y 24 dB.")
    channels = 1 if original.ndim == 1 else original.shape[0]
    result = original.astype(np.float32, copy=True)
    matrix = result[np.newaxis, :] if result.ndim == 1 else result
    gain_max = 10 ** (max_gain_db / 20)
    for (start, end), clip in zip(intervals, clips):
        fitted = _match_channels(fit_clip(clip, end - start, sr, mode=fit), channels)
        reference = matrix[:, start:end].copy()
        if rms_match:
            source_rms = float(np.sqrt(np.mean(np.square(fitted.astype(np.float64)))))
            reference_rms = float(np.sqrt(np.mean(np.square(reference.astype(np.float64)))))
            if reference_rms > 1e-7 and source_rms > 1e-7:
                fitted = fitted * np.clip(reference_rms / source_rms, 1 / gain_max, gain_max)
        # Fade the replacement into the original edge inside the interval. This
        # keeps the boundary continuous, and introduces no voice outside it.
        fade = min(round(sr * fade_ms / 1000), (end - start) // 2)
        if fade > 0:
            envelope = np.ones(end - start, dtype=np.float32)
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            envelope[:fade] = ramp
            envelope[-fade:] = ramp[::-1]
            fitted = fitted * envelope + reference * (1 - envelope)
        matrix[:, start:end] = fitted
    if not np.isfinite(result).all():
        raise ValueError("La reconstrucción produjo valores no finitos.")
    return result
