"""Zero-training spatial reasoning over a hand-authored 3D room.

The pipeline is deliberately perception-first: a room is authored as geometry,
scanned as images, reconstructed as a semantic point cloud by Gemma's own
vision encoder, segmented without supervision, and only then named -- by asking
Gemma what it sees in a cropped view.  Nothing the author typed reaches the
model, and no weights are trained on this machine.
"""

from __future__ import annotations

__all__ = ["room_spec"]
