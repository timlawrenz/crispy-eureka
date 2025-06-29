# When this custom node is loaded, import the hijack module to patch KSamplerAdvanced
from . import KSamplerNodes

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import nodes and add them to the mappings
try:
    from .ClassifierLoader import NODE_CLASS_MAPPINGS as CL_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as CL_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(CL_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(CL_DISPLAY_MAPPINGS)
except ImportError as e:
    print(f"[ClassifierGuidance] Could not import ClassifierLoader node: {e}")

try:
    from .ClassifierGuidance import NODE_CLASS_MAPPINGS as CG_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as CG_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(CG_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(CG_DISPLAY_MAPPINGS)
except ImportError as e:
    print(f"[ClassifierGuidance] Could not import ClassifierGuidance node: {e}")


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
