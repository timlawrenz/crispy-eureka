# When this custom node is loaded, import the hijack module to patch KSamplerAdvanced
from . import KSamplerNodes

# Import the new node and add it to the node mappings
try:
    from .ClassifierGuidance import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
except ImportError:
    print("Could not import ClassifierGuidance node.")
