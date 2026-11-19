# ComfyUI BigFans Audio

**Versión 0.1.1 · Dos voces con sus tiempos originales.**

Dos nodos para preparar y reconstruir conversaciones en ComfyUI: detectar quién habla por turnos o separar dos voces solapadas, enviar cada intervención a un conversor de voz y devolverla a su posición en el audio. Cada voz puede conservarse o convertirse de forma independiente.

El paquete procesa el audio en el servidor donde se ejecuta ComfyUI. No incluye pesos ni un motor de clonación de voz. Para convertir las voces se conecta un conversor externo, como Chatterbox mediante [TTS Audio Suite](https://github.com/diodiogod/TTS-Audio-Suite).

**Estado:** 75 tests aprobados con señales sintéticas, DSP real y adaptadores de inferencia simulados. Pendiente de validar inferencia con pesos reales e instalación completa en ComfyUI/RunningHub. Consulta [TESTING.md](TESTING.md).

## Instalación

Desde la carpeta de ComfyUI, usando su entorno Python:

```sh
cd custom_nodes
git clone https://github.com/MaxNotorio/ComfyUI-BigFans-Audio.git
cd ComfyUI-BigFans-Audio
python -m pip install -r requirements.txt
```

Reinicia ComfyUI y busca **BigFans → Audio**. En un servicio administrado, el proveedor debe instalar el repositorio y sus dependencias.

La instalación de dependencias debe hacerse en un entorno compatible: Pyannote requiere un conjunto reciente de Torch, Torchaudio y TorchCodec, y `pip` puede resolver o actualizar sus dependencias transitivas. El administrador debe revisar ese conjunto antes de instalar en un servidor compartido.

Los identificadores `H3TwoVoicePrepare` y `H3TwoVoiceRebuild` se conservan para admitir JSON existentes. **No cargues este paquete a la vez que ComfyUI-H3-TwoVoice**, porque ambos registran los mismos identificadores.

## Qué contiene

| Nodo visible | Función |
|---|---|
| **BigFans · Dos voces · detectar o separar** | Recibe un AUDIO; devuelve las intervenciones de A y B como listas, un plan temporal, dos preescuchas completas y un informe. |
| **BigFans · Dos voces · reconstruir conversación** | Recibe el plan y las dos listas; ajusta cada intervención a su duración y la coloca en su intervalo original. |

### Preparar la conversación

- **Modo 5 — turnos:** detecta dos hablantes mediante Community-1 o utiliza intervalos manuales. Recorta las intervenciones a partir del audio original. No separa voces que hablan a la vez.
- **Modo 6 — solapamientos:** separa el clip completo en dos pistas mediante SepFormer. Detecta actividad por energía RMS o utiliza intervalos manuales. Necesita el separador también cuando introduces los tiempos a mano.
- **`swap_ab`:** intercambia las pistas y sus intervalos. Escucha A y B antes de asignar las muestras vocales.
- **`device`:** CPU o CUDA para los modelos de análisis/separación.
- **`activity_db`:** umbral relativo de actividad de las pistas separadas; solo afecta al modo 6 automático.

A y B son etiquetas de pista: no indican género, rostro ni una identidad persistente entre clips.

`manual_segments` admite una intervención por línea, con tiempos en segundos:

```text
A,0.0,2.4
B,2.6,5.0
```

Los intervalos deben estar dentro del clip y contener actividad suficiente de ambas voces. En modo 5, los solapamientos de hasta 40 ms se dividen por el punto medio; los mayores se rechazan para utilizar el modo 6.

### Convertir y reconstruir

1. Conecta **intervenciones_A** a un conversor de voz con la muestra de A, e **intervenciones_B** a otro con la muestra de B. ComfyUI aplica el conversor a cada elemento de la lista.
2. Conecta las dos listas resultantes a **chunks_a** y **chunks_b** del nodo de reconstrucción. Conecta también **plan_temporal** directamente desde el nodo de preparación.
3. Selecciona **1 conservar** o **2 convertir** para cada voz. Para conservar una voz, conecta su lista original y evita ejecutar su conversor mediante la selección de ramas del workflow.

No concatenes las intervenciones antes de convertirlas. La reconstrucción necesita una salida por cada intervalo y en el mismo orden.

Si ambas voces se conservan, la salida es la fuente original, con sus canales y ambiente. Si alguna se convierte, la salida es una mezcla mono de duración original, con reducción de nivel si hace falta evitar recorte. El nodo remuestrea, ajusta la duración sin un cambio deliberado de tono y aplica transiciones cortas dentro de cada intervalo.

**`keep_residual` está desactivado inicialmente.** Al activarlo, se añade la diferencia entre la fuente y las pistas. Esa diferencia puede contener restos de voces originales; no es una separación garantizada de música o ambiente.

## Dependencias y modelos

- Python 3.10 o posterior; el entorno de tests utiliza Python 3.12.14.
- NumPy, SciPy y librosa según [requirements.txt](requirements.txt). Versión de librosa probada: **1.0.1**.
- `pyannote.audio==4.0.7` para diarización automática. Su manifiesto requiere Torch/Torchaudio ≥2.8 y TorchCodec ≥0.7, además de las bibliotecas FFmpeg correspondientes.
- `speechbrain==1.1.2` para separación. El adaptador utiliza `SepformerSeparation.separate_batch`.
- Torch/Torchaudio compatibles con el servidor ComfyUI. La versión de TorchCodec debe corresponder a Torch según la [matriz oficial](https://github.com/pytorch/torchcodec#installing-torchcodec).
- Un conversor externo de voz si deseas sustituir voces; no es necesario para conservarlas.

Las bibliotecas de inferencia se importan al ejecutar el modo correspondiente. Registrar los nodos no descarga modelos ni ejecuta inferencia. El modo 5 con intervalos manuales no carga Pyannote.

| Función | Modelo |
|---|---|
| Diarización automática | [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) |
| Separación de dos voces | [speechbrain/sepformer-whamr16k](https://huggingface.co/speechbrain/sepformer-whamr16k) |

Community-1 requiere aceptar sus condiciones de acceso en Hugging Face. Configura `HF_TOKEN` en el entorno del servidor cuando sea necesario, nunca dentro del workflow. El código no registra el token.

La caché predeterminada está en `ComfyUI/models/h3_two_voice/`. En la primera ejecución, los backends pueden descargar los modelos que falten. Para utilizar copias locales completas, configura antes de iniciar ComfyUI:

- `H3_TWO_VOICE_DIARIZATION_MODEL`: directorio de Community-1 con `config.yaml` y sus pesos dependientes resueltos.
- `H3_TWO_VOICE_SEPARATOR_MODEL`: directorio del separador con `hyperparams.yaml` y pesos.

Un directorio incompleto no permite ejecución sin conexión. Las licencias de los modelos son independientes de la licencia de este código.

## Límites

- Un clip por ejecución, de **0,5 a 15,5 segundos**, con dos voces y suficiente actividad. Hasta 8 canales de entrada y frecuencias de 8–192 kHz; el análisis utiliza mono.
- SepFormer procesa a **16 kHz**. Volver a la frecuencia original no recupera los agudos perdidos.
- La separación puede producir fugas, artefactos o intercambios de identidad. El clip se procesa completo para evitar intercambios entre ventanas; esto puede aumentar el consumo de memoria.
- Se rechazan conversiones vacías, silenciosas, no finitas o con cambios de duración excesivos. El ajuste temporal acepta una proporción de duración de 0,5–2 respecto al intervalo.
- Conservar los intervalos no garantiza alineación exacta de cada fonema ni sincronización labial. El generador de vídeo y cualquier etapa de sincronización posterior son componentes externos.
- No se han medido todavía calidad de separación, semejanza vocal, tiempos de GPU ni compatibilidad conjunta con los demás nodos de un servidor real.

## Tests

En un entorno con las dependencias DSP y pytest, sin descargar modelos:

```sh
python -m pip install 'numpy>=1.26,<3' 'scipy>=1.11,<2' 'librosa>=0.10.2,<2' pytest
python -m pytest -q --import-mode=importlib --rootdir=.. tests
```

Los tests comprueban tiempos, tono aproximado, remuestreo, canales, reconstrucción, listas de ComfyUI y errores de los adaptadores. Los modelos de diarización/separación y el transporte Torch están simulados. El informe reproducible y sus límites están en [TESTING.md](TESTING.md).

## Referencias y licencia

Contratos de integración revisados el 2026-11-19:

- [Pyannote 4.0.7 — Pipeline](https://github.com/pyannote/pyannote-audio/blob/4.0.7/src/pyannote/audio/core/pipeline.py).
- [SpeechBrain 1.1.2 — separación](https://github.com/speechbrain/speechbrain/blob/v1.1.2/speechbrain/inference/separation.py).
- [Configuración de SepFormer WHAMR16k](https://huggingface.co/speechbrain/sepformer-whamr16k/blob/main/hyperparams.yaml).
- [TTS Audio Suite — conversor de voz](https://github.com/diodiogod/TTS-Audio-Suite/blob/main/nodes/unified/voice_changer_node.py).

Código propio bajo [licencia MIT](LICENSE), Copyright © 2026 BigFans. No se incluyen pesos, grabaciones, workflows de terceros ni código de sus proyectos.
