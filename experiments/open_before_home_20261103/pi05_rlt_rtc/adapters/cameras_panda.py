"""Use exactly the accepted collecting pipeline's D405 preprocessing and clocks."""
from .robolab import activate
from .cameras import CameraFrame

def create_provider(config=None, **kwargs):
    activate()
    from pi05_demo.cameras import Pair
    cfg=dict(config or kwargs)
    cfg['views']=['wrist']
    pair=Pair(cfg)
    class Adapter:
        def capture(self):
            frames,metadata=pair.capture_with_metadata()
            return {k:CameraFrame(rgb,stamp,metadata[k]) for k,(rgb,stamp) in frames.items()}
        def close(self):
            pair.close()
    return Adapter()
