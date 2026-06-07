"""Pure-logic tests for the resolution service (DESIGN §5.4)."""

from __future__ import annotations

import pytest
from app.services import resolution


def test_snap16() -> None:
    assert resolution.snap16(682) == 688  # rounds 682.7 region up to 688
    assert resolution.snap16(688) == 688
    assert resolution.snap16(1) == 16  # never below 16
    assert resolution.snap16(1024) == 1024


def test_spec_example_landscape_3_2() -> None:
    # DESIGN §5.4 worked example: 1024 / landscape / 3:2 -> (1024, 688).
    assert resolution.resolve(base=1024, orientation="landscape", aspect="3:2") == (1024, 688)


def test_square_uses_base_for_both_sides() -> None:
    assert resolution.resolve(base=1024, orientation="square", aspect=None) == (1024, 1024)
    assert resolution.resolve(base=512, orientation="square", aspect="1:1") == (512, 512)


def test_portrait_longer_edge_is_height() -> None:
    w, h = resolution.resolve(base=1024, orientation="portrait", aspect="2:3")
    assert h == 1024  # longer edge is the height in portrait
    assert w < h
    assert w % 16 == 0 and h % 16 == 0


def test_landscape_16_9() -> None:
    w, h = resolution.resolve(base=1024, orientation="landscape", aspect="16:9")
    assert w == 1024
    assert h == resolution.snap16(int(round(1024 * 9 / 16)))  # 576
    assert h == 576


def test_aspect_order_independent() -> None:
    # "3:2" vs "2:3" both normalize to long:short for the orientation.
    land = resolution.resolve(base=1024, orientation="landscape", aspect="2:3")
    assert land == (1024, 688)


def test_match_source_snaps_to_16() -> None:
    w, h = resolution.resolve(
        base="match", orientation="square", aspect=None, source_dims=(1000, 700)
    )
    assert (w, h) == (resolution.snap16(1000), resolution.snap16(700))
    assert w % 16 == 0 and h % 16 == 0


def test_match_source_caps_long_edge() -> None:
    w, h = resolution.resolve(
        base="match",
        orientation="landscape",
        aspect=None,
        source_dims=(4000, 2000),
        max_long_edge=2048,
    )
    assert max(w, h) <= 2048
    assert w % 16 == 0 and h % 16 == 0


def test_match_without_source_falls_back_to_default() -> None:
    # Generate mode: no input -> falls back to default base square.
    assert resolution.resolve(base="match", orientation="square", aspect=None) == (
        resolution.DEFAULT_BASE_SIZE,
        resolution.DEFAULT_BASE_SIZE,
    )


def test_invalid_orientation_raises() -> None:
    with pytest.raises(ValueError):
        resolution.resolve(base=1024, orientation="diagonal", aspect="1:1")


def test_missing_aspect_for_landscape_raises() -> None:
    with pytest.raises(ValueError):
        resolution.resolve(base=1024, orientation="landscape", aspect=None)


def test_malformed_aspect_raises() -> None:
    with pytest.raises(ValueError):
        resolution.resolve(base=1024, orientation="landscape", aspect="not-an-aspect")


def test_enumerate_presets_shape() -> None:
    presets = resolution.enumerate_presets()
    assert presets["base_sizes"] == [512, 1024, 1536, 2048]
    assert presets["orientations"] == ["square", "portrait", "landscape"]
    assert presets["aspect_ratios"]["landscape"] == ["5:4", "4:3", "3:2", "16:9"]
    assert presets["snap_multiple"] == 16
    assert presets["match_supported"] is True
