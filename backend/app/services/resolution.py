"""Resolution preset service (DESIGN §5.4).

Output size is composed from three independent controls — **base size**, **orientation**,
and **aspect ratio** — into a concrete ``W×H``. This module is **pure** (no torch / no I/O)
so the math is trivially unit-testable and shared by the resolution router and the model
advisor (which needs the longer edge to estimate latent VRAM).

Rules (DESIGN §5.4):

- **Base size = the longer edge** (and the side length for Square). The shorter edge is
  derived from the chosen aspect ratio.
- Both dimensions are snapped to a multiple of **16** (latent-friendly) after derivation.
- **Match source** (Edit mode) uses the first input image's dimensions (snapped to ×16),
  optionally capped to a max long edge to protect VRAM. In Generate mode "Match source" is
  unavailable (no input), so it falls back to a default base size.

Verified example: ``1024 / landscape / 3:2`` → ``(1024, 688)`` (688 snapped from 682.7).
"""

from __future__ import annotations

# Base sizes offered in the picker (the longer-edge length). "match" is handled separately.
BASE_SIZES: list[int] = [512, 1024, 1536, 2048]

# Fallback base size when "match" is requested without source dims (Generate mode).
DEFAULT_BASE_SIZE: int = 1024

ORIENTATIONS: list[str] = ["square", "portrait", "landscape"]

# Aspect ratios offered per orientation (DESIGN §5.4 table). Expressed as "long:short"
# for landscape, "short:long" for portrait, and "1:1" for square.
ASPECT_RATIOS: dict[str, list[str]] = {
    "square": ["1:1"],
    "portrait": ["4:5", "3:4", "2:3", "9:16"],
    "landscape": ["5:4", "4:3", "3:2", "16:9"],
}


def snap16(x: int) -> int:
    """Snap a dimension to the nearest positive multiple of 16 (latent-friendly).

    Always returns at least 16 so a degenerate input can never yield a zero dimension.
    """
    snapped = int(round(x / 16.0)) * 16
    return max(16, snapped)


def _parse_aspect(aspect: str) -> tuple[int, int]:
    """Parse an ``"a:b"`` aspect string into a ``(a, b)`` integer pair."""
    try:
        a_str, b_str = aspect.split(":", 1)
        a, b = int(a_str), int(b_str)
    except (ValueError, AttributeError) as exc:  # malformed token
        raise ValueError(f"Invalid aspect ratio: {aspect!r}") from exc
    if a <= 0 or b <= 0:
        raise ValueError(f"Aspect ratio components must be positive: {aspect!r}")
    return a, b


def resolve(
    *,
    base: int | str,
    orientation: str,
    aspect: str | None,
    source_dims: tuple[int, int] | None = None,
    max_long_edge: int | None = None,
) -> tuple[int, int]:
    """Resolve a preset into a concrete ``(width, height)`` pair.

    Args:
        base: A longer-edge length (one of :data:`BASE_SIZES`, though any positive int is
            accepted) or the literal string ``"match"`` (Edit mode only).
        orientation: One of :data:`ORIENTATIONS`.
        aspect: An ``"a:b"`` aspect ratio. Required for portrait/landscape; ignored for
            square. May be ``None`` for square.
        source_dims: ``(width, height)`` of the first input image; used when ``base`` is
            ``"match"``.
        max_long_edge: Optional cap on the longer edge when matching source (protects VRAM).

    Returns:
        A ``(width, height)`` tuple, both snapped to a multiple of 16.

    Raises:
        ValueError: On an unknown orientation, a missing aspect for a non-square
            orientation, or a malformed aspect string.
    """
    if orientation not in ORIENTATIONS:
        raise ValueError(f"Unknown orientation: {orientation!r}")

    # --- Match source (Edit mode): preserve the source dimensions, snapped to 16. ---
    if isinstance(base, str) and base.lower() == "match":
        if source_dims is None:
            # Generate mode has no input — fall back to a square default base.
            return snap16(DEFAULT_BASE_SIZE), snap16(DEFAULT_BASE_SIZE)
        src_w, src_h = source_dims
        if src_w <= 0 or src_h <= 0:
            raise ValueError(f"Invalid source dimensions: {source_dims!r}")
        if max_long_edge is not None and max_long_edge > 0:
            long_edge = max(src_w, src_h)
            if long_edge > max_long_edge:
                scale = max_long_edge / long_edge
                src_w = int(round(src_w * scale))
                src_h = int(round(src_h * scale))
        return snap16(src_w), snap16(src_h)

    if isinstance(base, str):
        # Accept a numeric string (clients often send select values as strings, e.g. "1024");
        # only non-numeric, non-"match" strings are invalid.
        if not base.strip().isdigit():
            raise ValueError(f"Unknown base size: {base!r}")
        base = int(base)

    long_edge = int(base)
    if long_edge <= 0:
        raise ValueError(f"Base size must be positive: {base!r}")

    # --- Square: base is the side length for both dimensions. ---
    if orientation == "square":
        side = snap16(long_edge)
        return side, side

    # --- Portrait / landscape: derive the short edge from the aspect ratio. ---
    if aspect is None:
        raise ValueError(f"Aspect ratio is required for orientation {orientation!r}")
    a, b = _parse_aspect(aspect)
    # ``a:b`` is conventionally long:short. Normalize so ratio >= 1.
    long_ratio, short_ratio = (a, b) if a >= b else (b, a)
    short_edge = long_edge * short_ratio / long_ratio

    if orientation == "landscape":
        return snap16(long_edge), snap16(int(round(short_edge)))
    # portrait — the longer edge is the height.
    return snap16(int(round(short_edge))), snap16(long_edge)


def enumerate_presets() -> dict[str, object]:
    """Return the preset enumerations for the API (DESIGN §5.4 / §7).

    Includes ``"match"`` as a selectable base alongside the numeric sizes so the frontend
    can render the full picker.
    """
    return {
        "base_sizes": list(BASE_SIZES),
        "match_supported": True,
        "default_base_size": DEFAULT_BASE_SIZE,
        "orientations": list(ORIENTATIONS),
        "aspect_ratios": {k: list(v) for k, v in ASPECT_RATIOS.items()},
        "snap_multiple": 16,
    }
