"""Lazy inference backends for the H3 two-voice bridge.

Audited contracts: pyannote.audio 4.0.7 / Community-1 and SpeechBrain 1.1.2.
No weights or inference packages are loaded when ComfyUI registers these nodes.
Speaker labels are anonymous; neither backend classifies gender or face identity.
"""

from __future__ import annotations

import gc
import hashlib
import math
import os
from pathlib import Path
import re

import numpy as np


ANALYSIS_RATE = 16000
MAX_SEPARATOR_SECONDS = 15.5
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
SEPARATOR_MODEL = "speechbrain/sepformer-whamr16k"


def _audio(mono: np.ndarray, sr: int) -> np.ndarray:
    if isinstance(sr, bool) or not isinstance(sr, (int, np.integer)) or sr <= 0:
        raise ValueError("La frecuencia de muestreo debe ser un entero positivo.")
    value = np.asarray(mono)
    if value.ndim != 1 or value.size == 0 or np.iscomplexobj(value):
        raise ValueError("Se necesita un audio mono no vacío, con forma [muestras].")
    try:
        value = np.ascontiguousarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("El audio debe contener muestras numéricas finitas.") from None
    if not np.isfinite(value).all():
        raise ValueError("El audio contiene muestras NaN o infinitas.")
    return value


def _fit_length(audio: np.ndarray, length: int) -> np.ndarray:
    """Correct only end padding/rounding, never rescale peaks or shift content."""
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < length:
        audio = np.pad(audio, (0, length - len(audio)))
    return np.ascontiguousarray(audio[:length], dtype=np.float32)


def _resample(audio: np.ndarray, src: int, dst: int, length: int | None = None) -> np.ndarray:
    target = max(1, round(len(audio) * dst / src)) if length is None else length
    if src == dst:
        return _fit_length(audio, target)
    try:
        from scipy.signal import resample_poly
    except Exception:
        raise RuntimeError("Falta SciPy o no se puede importar. Instala las dependencias del nodo para remuestrear el audio.") from None
    divisor = math.gcd(int(src), int(dst))
    try:
        converted = resample_poly(audio, dst // divisor, src // divisor)
    except Exception:
        raise RuntimeError("No se pudo remuestrear el audio con SciPy. Comprueba la instalación y la memoria disponible.") from None
    return _fit_length(converted, target)


def _torch():
    try:
        import torch
        return torch
    except Exception:
        raise RuntimeError("No se puede importar PyTorch. Revisa las dependencias del nodo en el entorno de ComfyUI.") from None


def _device(torch, requested: str):
    name = str(requested).strip().lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name != "cpu" and not re.fullmatch(r"cuda(?::\d+)?", name):
        raise ValueError("Dispositivo no admitido. Elige auto, cpu, cuda o cuda:N.")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible para este backend. Elige cpu o revisa la instalación de PyTorch/CUDA.")
    if ":" in name and int(name.split(":")[1]) >= torch.cuda.device_count():
        raise ValueError("La GPU seleccionada no existe. Elige una GPU disponible o cpu.")
    return torch.device(name)


def _source(env_name: str, default: str, config_name: str) -> str:
    override = os.environ.get(env_name, "").strip()
    if not override:
        return default
    local = Path(override).expanduser()
    valid = (local.is_dir() and (local / config_name).is_file())
    if env_name == "H3_TWO_VOICE_DIARIZATION_MODEL":
        valid = valid or (local.is_file() and local.name == config_name)
    if not valid:
        raise RuntimeError(
            f"{env_name} debe señalar un modelo local completo con {config_name}. "
            "Corrige esa variable o elimínala para usar el modelo predeterminado."
        )
    return str(local.resolve())


def _cache(cache_dir: str, backend: str, source: str) -> str:
    try:
        # Distinct local overrides must not accidentally reuse each other's files.
        key = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
        directory = Path(cache_dir).expanduser() / backend / key
        directory.mkdir(parents=True, exist_ok=True)
        return str(directory)
    except Exception:
        raise RuntimeError("No se puede crear la caché de modelos. Configura una carpeta con permiso de escritura y espacio libre.") from None


def _release(model, torch, device) -> None:
    """Release GPU allocations even after inference fails; never mask its error."""
    if model is not None:
        try:
            model.to(torch.device("cpu"))
        except Exception:
            pass
    gc.collect()
    if str(device).startswith("cuda"):
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def diarize_turns(mono: np.ndarray, sr: int, device: str, cache_dir: str) -> list[dict]:
    """Return regular (possibly overlapping) timestamps for exactly two speakers.

    This does not isolate overlapping speech. The caller must assess overlap and
    route it to source separation instead of treating the cuts as clean voices.
    """
    original = _audio(mono, sr)
    waveform = _resample(original, int(sr), ANALYSIS_RATE)
    torch = _torch()
    selected_device = _device(torch, device)
    try:
        from pyannote.audio import Pipeline
    except Exception:
        raise RuntimeError(
            "No se puede importar pyannote.audio. Este nodo está preparado para pyannote.audio 4.0.7; "
            "revisa también sus dependencias PyTorch, torchaudio y torchcodec con el administrador."
        ) from None
    source = _source("H3_TWO_VOICE_DIARIZATION_MODEL", DIARIZATION_MODEL, "config.yaml")
    cache = _cache(cache_dir, "pyannote", source)
    model = None
    try:
        try:
            model = Pipeline.from_pretrained(source, token=os.environ.get("HF_TOKEN") or None, cache_dir=cache)
            if model is None:
                raise RuntimeError("Pipeline not returned")
            model.to(selected_device)
        except Exception:
            raise RuntimeError(
                "No se pudo cargar Community-1. Acepta sus condiciones en Hugging Face "
                "(pyannote/speaker-diarization-community-1, licencia CC-BY-4.0) y configura HF_TOKEN "
                "en el servidor si la descarga lo requiere, o usa H3_TWO_VOICE_DIARIZATION_MODEL "
                "con una copia local completa. Comprueba también espacio, memoria y dependencias."
            ) from None
        try:
            with torch.inference_mode():
                result = model(
                    {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": ANALYSIS_RATE},
                    num_speakers=2,
                )
            annotation = result.speaker_diarization
            records = list(annotation.itertracks(yield_label=True))
        except Exception:
            raise RuntimeError(
                "Community-1 no pudo producir diarización regular. Comprueba pyannote.audio 4.0.7, "
                "la memoria disponible y que el clip contenga voz audible. No se utiliza la salida exclusive."
            ) from None
        duration = len(original) / sr
        turns = []
        for turn, _, label in records:
            try:
                start, end = float(turn.start), float(turn.end)
            except (AttributeError, TypeError, ValueError):
                raise RuntimeError("El backend devolvió marcas temporales de voz inválidas.") from None
            if not math.isfinite(start) or not math.isfinite(end) or end < start:
                raise RuntimeError("El backend devolvió marcas temporales de voz inválidas.")
            start, end = max(0.0, start), min(duration, end)
            if end > start:
                turns.append({"speaker": str(label), "start": start, "end": end})
        if len({item["speaker"] for item in turns}) != 2:
            raise RuntimeError(
                "No se han identificado exactamente dos hablantes con turnos útiles. "
                "Usa un clip con voz clara de ambas personas; los clips muy cortos pueden no bastar."
            )
        return sorted(turns, key=lambda item: (item["start"], item["end"], item["speaker"]))
    finally:
        _release(model, torch, selected_device)


def separate_two(mono: np.ndarray, sr: int, device: str, cache_dir: str) -> list[np.ndarray]:
    """Separate a complete <=15.5 s clip, preserving sample count and level.

    Output order is arbitrary; listen to the stems before assigning voices.
    No chunking, per-stem peak normalization or guarantee of clean isolation.
    """
    original = _audio(mono, sr)
    if len(original) / sr > MAX_SEPARATOR_SECONDS:
        raise ValueError("La separación simultánea admite clips de hasta 15,5 segundos. Recorta el clip antes de ejecutarla.")
    waveform = _resample(original, int(sr), ANALYSIS_RATE)
    torch = _torch()
    selected_device = _device(torch, device)
    try:
        from speechbrain.inference.separation import SepformerSeparation
    except Exception:
        raise RuntimeError(
            "No se puede importar SpeechBrain. Este backend está preparado para SpeechBrain 1.1.2; "
            "revisa sus dependencias y compatibilidad con PyTorch/torchaudio."
        ) from None
    source = _source("H3_TWO_VOICE_SEPARATOR_MODEL", SEPARATOR_MODEL, "hyperparams.yaml")
    cache = _cache(cache_dir, "sepformer", source)
    model = None
    try:
        try:
            model = SepformerSeparation.from_hparams(
                source=source, savedir=cache, run_opts={"device": str(selected_device)}
            )
            if model is None:
                raise RuntimeError("Separator not returned")
        except Exception:
            raise RuntimeError(
                "No se pudo cargar SepFormer WHAMR 16 kHz. Comprueba acceso a "
                "speechbrain/sepformer-whamr16k, espacio y dependencias, o configura "
                "H3_TWO_VOICE_SEPARATOR_MODEL con una copia local completa."
            ) from None
        try:
            with torch.inference_mode():
                estimated = model.separate_batch(torch.from_numpy(waveform).unsqueeze(0))
                values = estimated.detach().float().cpu().numpy()
            del estimated
        except Exception:
            raise RuntimeError(
                "SepFormer no pudo separar el clip completo. Comprueba la memoria y las dependencias; "
                "prueba un clip más corto o el dispositivo cpu. No se trocea el audio para evitar intercambiar los hablantes."
            ) from None
        if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != 2 or values.shape[1] == 0:
            raise RuntimeError("SepFormer devolvió un formato inesperado; se necesitan dos voces con forma [1,muestras,2].")
        if not np.isfinite(values).all():
            raise RuntimeError("SepFormer devolvió muestras NaN o infinitas. Comprueba el modelo y el audio de entrada.")
        # separate_batch already restores T in the audited version. This also
        # handles harmless encoder padding from compatible implementations.
        stems = [_fit_length(values[0, :, index], len(waveform)) for index in (0, 1)]
        return [_resample(stem, ANALYSIS_RATE, int(sr), len(original)) for stem in stems]
    finally:
        _release(model, torch, selected_device)
