from . import pano3d, ldif, mgnet, multi_view, total3d
from . import loss

method_paths = {
    'Pano3D': pano3d,
    'LIENLDIF': ldif,
    'MGNet': mgnet,
    'MultiView': multi_view,
    'Total3D': total3d,
}

__all__ = ['method_paths']