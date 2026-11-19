"""Timestamp-preserving bridge around existing Chatterbox VC nodes.

Prepare produces lists of utterances. ComfyUI maps the normal voice changer
over each list. Rebuild receives whole lists and places them back in time.
No model inference or network access happens on node registration.
"""
from pathlib import Path
import json
import numpy as np

from . import audio_ops
from . import backends

MAX_SECONDS = 15.5
PLAN_TYPE = "H3_VOICE_PLAN"


def _read_audio(audio):
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise ValueError("Se esperaba AUDIO de ComfyUI, con waveform y sample_rate.")
    sr = audio.get("sample_rate")
    if isinstance(sr, bool) or not isinstance(sr, (int, np.integer)) or not 8000 <= sr <= 192000:
        raise ValueError("La frecuencia del audio debe estar entre 8000 y 192000 Hz.")
    wave = audio["waveform"]
    if hasattr(wave, "detach"):
        wave = wave.detach().cpu().float().numpy()
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim == 3:
        if wave.shape[0] != 1:
            raise ValueError("Dos voces admite un clip por ejecución, no un batch de clips.")
        wave = wave[0]
    elif wave.ndim == 1:
        wave = wave[None, :]
    if wave.ndim != 2 or not 1 <= wave.shape[0] <= 8 or wave.shape[-1] == 0:
        raise ValueError("Audio vacío o forma no admitida; se requiere [1, canales, muestras].")
    if not np.isfinite(wave).all():
        raise ValueError("El audio contiene valores NaN o infinitos.")
    return np.array(wave, dtype=np.float32, copy=True), int(sr)


def _audio(wave, sr):
    # torch belongs to the ComfyUI host, not to this package's installer.
    import torch
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim == 1:
        wave = wave[None, :]
    return {"waveform": torch.from_numpy(np.ascontiguousarray(wave[None])).contiguous(),
            "sample_rate": int(sr)}


def _cache_dir():
    import folder_paths
    return str(Path(folder_paths.models_dir) / "h3_two_voice")


def _scalar(values, name):
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(f"{name}: se esperaba un único valor compartido.")
    return values[0]


def _resample_to(wave, from_sr, to_sr):
    return audio_ops.resample_audio(wave, from_sr, to_sr).astype(np.float32)


class H3TwoVoicePrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "mode": ("INT", {"default": 5, "min": 5, "max": 6,
                              "tooltip": "5: dos voces por turnos. 6: dos voces solapadas."}),
            "swap_ab": ("BOOLEAN", {"default": False}),
            "activity_db": ("FLOAT", {"default": -40.0, "min": -60.0, "max": -15.0, "step": 1.0,
                                      "tooltip": "Umbral relativo de actividad en las pistas separadas; solo modo 6."}),
            "manual_segments": ("STRING", {"default": "", "multiline": True,
                "tooltip": "Opcional: una fila A,inicio,final o B,inicio,final en segundos. Vacío = automático."}),
            "device": (["cpu", "cuda"], {"default": "cpu"}),
        }}

    RETURN_TYPES = ("AUDIO", "AUDIO", PLAN_TYPE, "AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("intervenciones_A", "intervenciones_B", "plan_temporal", "escuchar_A", "escuchar_B", "informe")
    OUTPUT_IS_LIST = (True, True, False, False, False, False)
    FUNCTION = "prepare"
    CATEGORY = "BigFans/Audio"

    def prepare(self, audio, mode=5, swap_ab=False, activity_db=-40.0, manual_segments="", device="cpu"):
        if mode not in (5, 6):
            raise ValueError("Este bloque solo se utiliza en los modos de audio 5 o 6.")
        if device not in ("cpu", "cuda"):
            raise ValueError("El dispositivo debe ser cpu o cuda.")
        source, sr = _read_audio(audio)
        n = source.shape[-1]
        if not 0.5 <= n / sr <= MAX_SECONDS:
            raise ValueError("Usa un clip de 0,5 a 15,5 segundos. El workflow H3 ofrece 3–15 segundos nominales.")
        mono = source.mean(axis=0)
        if float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))) < 1e-6:
            raise ValueError("La fuente está en silencio. Comprueba el audio H3 o la grabación cargada.")
        manual = str(manual_segments or "").strip()
        if mode == 5:
            if manual:
                rows = audio_ops.parse_manual_segments(manual, sr, n, reject_overlap=True, relabel=False)
            else:
                detected = backends.diarize_turns(mono, sr, device, _cache_dir())
                rows = audio_ops.segments_from_seconds(detected, sr, n, reject_overlap=True, relabel=True)
            if {r["speaker"] for r in rows} != {"A", "B"}:
                raise ValueError("No se han obtenido dos hablantes. Usa un diálogo más claro o corrige los turnos manualmente.")
            tracks = {k: np.zeros(n, dtype=np.float32) for k in ("A", "B")}
            for r in rows:
                tracks[r["speaker"]][r["start"]:r["end"]] = mono[r["start"]:r["end"]]
            intervals = {k: [(r["start"], r["end"]) for r in rows if r["speaker"] == k] for k in ("A", "B")}
            if swap_ab:
                tracks = {"A": tracks["B"], "B": tracks["A"]}
                intervals = {"A": intervals["B"], "B": intervals["A"]}
        else:
            separated = backends.separate_two(mono, sr, device, _cache_dir())
            if len(separated) != 2:
                raise ValueError("El separador no devolvió exactamente dos pistas.")
            a, b = [np.asarray(t, dtype=np.float32).reshape(-1) for t in separated]
            if len(a) != n or len(b) != n or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("Las pistas separadas no mantienen el número de muestras original.")
            tracks = {"A": a, "B": b}
            if manual:
                rows = audio_ops.parse_manual_segments(manual, sr, n, reject_overlap=False, relabel=False)
                if {r["speaker"] for r in rows} != {"A", "B"}:
                    raise ValueError("La corrección manual debe incluir intervalos para A y para B.")
                intervals = {k: [(r["start"], r["end"]) for r in rows if r["speaker"] == k] for k in ("A", "B")}
            else:
                intervals = {k: audio_ops.activity_segments(tracks[k], sr, threshold_db=float(activity_db)) for k in ("A", "B")}
            if swap_ab:
                tracks = {"A": tracks["B"], "B": tracks["A"]}
                intervals = {"A": intervals["B"], "B": intervals["A"]}
        for k in ("A", "B"):
            if not intervals[k] or sum(e - s for s, e in intervals[k]) < round(0.2 * sr):
                raise ValueError(f"La pista {k} no contiene suficiente actividad. Escucha la fuente y revisa la detección o los intervalos.")
        plan = {"version": 1, "sample_rate": sr, "sample_count": n, "mode": int(mode),
                "source": source, "tracks": tracks, "segments": intervals,
                "residual": mono - tracks["A"] - tracks["B"]}
        chunks = {k: [_audio(tracks[k][s:e], sr) for s, e in intervals[k]] for k in ("A", "B")}
        listing = {k: [[round(s / sr, 3), round(e / sr, 3)] for s, e in intervals[k]] for k in ("A", "B")}
        report = (f"Modo {mode}: {'turnos' if mode == 5 else 'separación de voces solapadas'}\n"
                  f"Duración: {n / sr:.3f} s · {sr} Hz · intercambio A/B: {bool(swap_ab)}\n"
                  "Intervalos [inicio, final] en segundos:\n" + json.dumps(listing, ensure_ascii=False, indent=2) +
                  "\nA y B son etiquetas de pista, no género ni identidad visual. Escucha ambas antes de asignar las muestras." +
                  ("\nSepFormer procesa a 16 kHz. Volver a la frecuencia original no recupera los agudos perdidos." if mode == 6 else ""))
        return chunks["A"], chunks["B"], plan, _audio(tracks["A"], sr), _audio(tracks["B"], sr), report


class H3TwoVoiceRebuild:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": (PLAN_TYPE,), "chunks_a": ("AUDIO",), "chunks_b": ("AUDIO",),
            "action_a": ("INT", {"default": 2, "min": 1, "max": 2}),
            "action_b": ("INT", {"default": 2, "min": 1, "max": 2}),
            "keep_residual": ("BOOLEAN", {"default": False,
                 "tooltip": "Añade fuente menos pistas separadas. Puede reintroducir restos de las voces originales; inicialmente apagado."}),
        }}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("conversacion", "informe")
    INPUT_IS_LIST = True
    FUNCTION = "rebuild"
    CATEGORY = "BigFans/Audio"

    def rebuild(self, plan, chunks_a, chunks_b, action_a, action_b, keep_residual):
        p = _scalar(plan, "plan")
        actions = {"A": int(_scalar(action_a, "acción A")), "B": int(_scalar(action_b, "acción B"))}
        retain = bool(_scalar(keep_residual, "resto de audio"))
        if not isinstance(p, dict) or p.get("version") != 1:
            raise ValueError("Plan temporal ausente o incompatible; conecta el mismo bloque de preparación.")
        if any(x not in (1, 2) for x in actions.values()):
            raise ValueError("Cada voz debe seleccionar 1 conservar o 2 convertir.")
        sr, n = int(p["sample_rate"]), int(p["sample_count"])
        if actions == {"A": 1, "B": 1}:
            return _audio(p["source"], sr), "A y B conservadas: salida original sin procesar, incluidos canales y ambiente."
        result = {}
        for k, clips in (("A", chunks_a), ("B", chunks_b)):
            if not isinstance(clips, list) or len(clips) != len(p["segments"][k]):
                raise ValueError(f"La cantidad de intervenciones de {k} no coincide con el plan temporal.")
            converted = []
            for clip in clips:
                wave, clip_sr = _read_audio(clip)
                converted.append(_resample_to(wave.mean(axis=0), clip_sr, sr))
            result[k] = audio_ops.build_track(p["tracks"][k], sr, p["segments"][k], converted,
                                              action=actions[k], fit="stretch")
        mixed = np.asarray(result["A"] + result["B"], dtype=np.float32)
        if retain:
            mixed += p["residual"]
        if mixed.shape != (n,) or not np.isfinite(mixed).all():
            raise ValueError("La reconstrucción produjo una forma o valores inválidos.")
        peak = float(np.max(np.abs(mixed)))
        scale = min(1.0, 0.99 / peak) if peak > 0 else 1.0
        mixed *= scale
        report = (f"A: {'conservada' if actions['A'] == 1 else 'convertida'} · "
                  f"B: {'conservada' if actions['B'] == 1 else 'convertida'}\n"
                  f"Salida: {n} muestras, {n / sr:.3f} s, {sr} Hz. Resto: {retain}. Ganancia antirrecorte: {scale:.4f}.\n"
                  "Cada intervención se ajusta y recoloca en su intervalo. Esto conserva los turnos; no garantiza alineación exacta de cada fonema ni labios.")
        return _audio(mixed, sr), report
