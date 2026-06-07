"""Background removal / subject cutout service (DESIGN §13 #1).

One-click isolate a subject from an uploaded image before compositing — the small,
high-leverage tool that powers the furniture workflow: drop a chair photo, auto-cut it,
then place it into a pinned room (DESIGN §13 #1). The cutout is produced with **rembg**
(the U^2-Net-family matting models), returning an **RGBA** image whose background is
transparent so it drops cleanly onto another scene.

``rembg`` lives in the ``[inference]`` extra and is **not** present in the torch-free dev
image, so it is imported **lazily inside the function** — this module imports fine without
it, and only a real cutout call needs the model.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

from app.logging_utils import phase

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

log = logging.getLogger(__name__)


def remove_background(*, image: PILImage) -> PILImage:
    """Cut the subject out of ``image``, returning an RGBA image (DESIGN §13 #1).

    The returned image is the same size as the input with its background made
    transparent (alpha = 0), so it composites directly onto a pinned background.

    Args:
        image: The source PIL image (any mode; converted as needed).

    Returns:
        A new ``RGBA`` :class:`PIL.Image.Image` — the matted subject on transparency.

    Raises:
        RuntimeError: If ``rembg`` is not installed (it lives in the ``[inference]``
            extra; the torch-free dev image does not ship it).
    """
    from PIL import Image  # local import keeps the module light at import time

    try:
        from rembg import remove  # noqa: PLC0415 — lazy: rembg is an [inference]-only dep
    except ImportError as exc:  # pragma: no cover - exercised via the gpu-marked test
        raise RuntimeError(
            "Background removal needs 'rembg' (the [inference] extra); it is not installed "
            "in the dev image. Install the inference stack to use this tool."
        ) from exc

    with phase(log, f"Removing background ({image.width}x{image.height}) via rembg"):
        # rembg.remove accepts a PIL image and (with the default session) returns an RGBA
        # PIL image with the background alpha-masked out.
        result = remove(image)

    # Normalize to a guaranteed RGBA PIL image regardless of what rembg hands back
    # (it can return bytes when given bytes; PIL image in / PIL image out is the contract).
    if isinstance(result, (bytes, bytearray)):
        result = Image.open(io.BytesIO(bytes(result)))
    if result.mode != "RGBA":
        result = result.convert("RGBA")
    return result
