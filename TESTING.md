# Verificación de BigFans Audio 0.1.1

Fecha: **2026-11-19**.

Resultado: **75 tests aprobados** en 2,63 segundos, con 3 avisos de ejecución de Numba. No se descargaron modelos ni se ejecutó inferencia con pesos reales.

## Entorno utilizado

| Componente | Versión |
|---|---|
| Python | 3.12.14 |
| NumPy | 2.5.5 |
| SciPy | 1.18.1 |
| librosa | 1.0.1 |
| Numba | 0.68.1 |
| llvmlite | 0.50.0 |
| pytest | 9.2.0 |

Torch, Pyannote y SpeechBrain se sustituyeron por dobles de prueba en las comprobaciones de adaptadores y transporte de tensores. NumPy, SciPy y librosa ejecutaron el procesamiento DSP real.

Los pines de inferencia de `requirements.txt` corresponden a los contratos revisados de `pyannote.audio==4.0.7` y `speechbrain==1.1.2`. Esta ejecución de tests no demuestra que esas dependencias se instalen juntas correctamente en un servidor determinado.

## Reproducción

Desde la raíz de este repositorio, en un entorno de prueba con las dependencias DSP y pytest:

```sh
python -m pytest -q --import-mode=importlib --rootdir=.. -p no:cacheprovider tests
```

`--rootdir=..` permite a pytest importar el paquete ComfyUI con contexto de paquete, en lugar de cargar su `__init__.py` como un módulo sin padre. No cambia el registro de los nodos en ComfyUI.

## Cobertura

| Archivo | Tests | Qué comprueba |
|---|---:|---|
| `tests/test_audio_ops.py` | 27 | Validación y redondeo de tiempos, solapamientos, detección de actividad, remuestreo, ajuste de duración, tono aproximado, mezcla y conservación exacta. |
| `tests/test_backends.py` | 20 | Contratos de llamadas simuladas a Pyannote y SpeechBrain, formato y duración de salida, caché, limpieza tras errores y ausencia de credenciales en mensajes. |
| `tests/test_nodes.py` | 28 | Listas de ComfyUI, turnos y solapes, intercambio A/B, acciones independientes, fuente estéreo original al conservar ambas voces, remuestreo y rechazo de entradas inválidas. |

## Avisos observados

Numba emitió `RuntimeWarning: invalid value encountered in cast` en tres pruebas de ajuste temporal mediante librosa. Las salidas terminaron correctamente y pasaron las comprobaciones de muestras finitas, duración y frecuencia. Los avisos no se ocultaron. Conviene comprobar su comportamiento en el conjunto de versiones que utilice el servidor final.

## Pendiente de comprobar

- Instalación conjunta de ComfyUI, Torch/Torchaudio/TorchCodec y los backends de inferencia.
- Inferencia real con Community-1 y SepFormer, incluidos acceso a los pesos y uso de CPU/CUDA.
- Calidad de separación, posibles fugas de voces, conversión con Chatterbox y semejanza a las muestras vocales.
- Sincronización labial de una etapa de vídeo posterior, tiempos de GPU y consumo de memoria.

Las señales sintéticas validan el procesamiento y sus contratos; no representan una evaluación perceptual de conversaciones reales.
