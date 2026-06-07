"""Device detection endpoint (DESIGN §4.3, §7, §9.2).

Exposes the detected accelerator and its capabilities so the frontend can render the model
advisor UI and keep the resolution/precision pickers in sync.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.schemas.advisor import DeviceInfoRead
from app.services.device_manager import get_device_info

router = APIRouter(prefix="/api/device", tags=["device"])


@router.get("", response_model=DeviceInfoRead)
def read_device(refresh: bool = Query(default=False)) -> DeviceInfoRead:
    """Return the detected backend, SM/caps, total/free VRAM, and RAM (DESIGN §7).

    Args:
        refresh: Re-probe the device (free VRAM drifts as jobs run) instead of using the
            cached snapshot.
    """
    return DeviceInfoRead.from_info(get_device_info(refresh=refresh))
