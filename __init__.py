"""ComfyUI registration; model libraries are loaded only when needed."""
__version__ = "0.1.1"

from .nodes import H3TwoVoicePrepare, H3TwoVoiceRebuild

NODE_CLASS_MAPPINGS = {
    "H3TwoVoicePrepare": H3TwoVoicePrepare,
    "H3TwoVoiceRebuild": H3TwoVoiceRebuild,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3TwoVoicePrepare": "BigFans · Dos voces · detectar o separar",
    "H3TwoVoiceRebuild": "BigFans · Dos voces · reconstruir conversación",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
