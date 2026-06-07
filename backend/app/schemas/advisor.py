"""Pydantic request/response models for the device + model-advisor APIs (DESIGN §9, §7).

These shape the JSON surface of ``/api/device`` and ``/api/models`` and provide thin
serialization adapters over the frozen dataclasses returned by the services
(:class:`~app.services.device_manager.DeviceInfo`,
:class:`~app.services.model_advisor.Advice`).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.services.device_manager import DeviceInfo
from app.services.model_advisor import Advice, PrecisionOption


class DeviceInfoRead(BaseModel):
    """Public view of the detected accelerator (DESIGN §4.3, §9.2)."""

    backend: str
    device_str: str
    name: str
    compute_capability: tuple[int, int] | None = None
    total_vram_mb: int
    free_vram_mb: int
    total_ram_mb: int
    supports_bf16: bool
    supports_fp8_native: bool
    supports_int4_nunchaku: bool

    @classmethod
    def from_info(cls, info: DeviceInfo) -> DeviceInfoRead:
        """Build a read model from a :class:`DeviceInfo` dataclass."""
        return cls(
            backend=info.backend,
            device_str=info.device_str,
            name=info.name,
            compute_capability=info.compute_capability,
            total_vram_mb=info.total_vram_mb,
            free_vram_mb=info.free_vram_mb,
            total_ram_mb=info.total_ram_mb,
            supports_bf16=info.supports_bf16,
            supports_fp8_native=info.supports_fp8_native,
            supports_int4_nunchaku=info.supports_int4_nunchaku,
        )


class PrecisionOptionRead(BaseModel):
    """One ranked precision option (DESIGN §9.2)."""

    precision: str
    available: bool
    status: str
    est_peak_vram_mb: int
    headroom_mb: int
    rationale: str
    caveats: list[str]

    @classmethod
    def from_option(cls, opt: PrecisionOption) -> PrecisionOptionRead:
        return cls(
            precision=opt.precision,
            available=opt.available,
            status=opt.status,
            est_peak_vram_mb=opt.est_peak_vram_mb,
            headroom_mb=opt.headroom_mb,
            rationale=opt.rationale,
            caveats=list(opt.caveats),
        )


class AdviceRead(BaseModel):
    """Ranked advice for a planned run (DESIGN §9.2)."""

    device: DeviceInfoRead
    options: list[PrecisionOptionRead]
    recommended: str | None = None

    @classmethod
    def from_advice(cls, advice: Advice) -> AdviceRead:
        return cls(
            device=DeviceInfoRead.from_info(advice.device),
            options=[PrecisionOptionRead.from_option(o) for o in advice.options],
            recommended=advice.recommended,
        )


class ResolutionSpec(BaseModel):
    """A resolution preset selection, resolvable to a concrete ``W×H`` (DESIGN §5.4)."""

    base: int | str = Field(description='Longer-edge length, or the literal "match".')
    orientation: str = "square"
    aspect: str | None = None
    source_dims: tuple[int, int] | None = Field(
        default=None, description="(width, height) of the first input image, for 'match'."
    )
    max_long_edge: int | None = None


class LoraSelection(BaseModel):
    """A selected LoRA + weight (only the count matters to the advisor; DESIGN §9.2)."""

    lora_id: str
    weight: float = 1.0


class AdviseRequest(BaseModel):
    """Body for ``POST /api/models/advise`` (DESIGN §7, §9.2).

    Provide either an explicit ``longer_edge`` or a ``resolution`` spec (resolved via the
    resolution service). ``resolution`` takes precedence when both are given.
    """

    mode: str = "edit"
    longer_edge: int | None = None
    resolution: ResolutionSpec | None = None
    batch: int = 1
    loras: list[LoraSelection] = Field(default_factory=list)


class ModelOptionRead(BaseModel):
    """A configured model id with the precisions available on the current device."""

    mode: str
    model_id: str
    available_precisions: list[str]


class ModelsResponse(BaseModel):
    """Response for ``GET /api/models`` (DESIGN §7)."""

    device: DeviceInfoRead
    models: list[ModelOptionRead]
    default_precision: str


class ResolveRequest(BaseModel):
    """Body for ``POST /api/presets/resolution/resolve`` (DESIGN §5.4, §7)."""

    base: int | str
    orientation: str = "square"
    aspect: str | None = None
    source_dims: tuple[int, int] | None = None
    max_long_edge: int | None = None


class ResolveResponse(BaseModel):
    """Resolved concrete dimensions."""

    w: int
    h: int


class PresetsResponse(BaseModel):
    """Response for ``GET /api/presets/resolution`` (DESIGN §5.4)."""

    base_sizes: list[int]
    match_supported: bool
    default_base_size: int
    orientations: list[str]
    aspect_ratios: dict[str, list[str]]
    snap_multiple: int
