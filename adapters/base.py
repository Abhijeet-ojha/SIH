"""
adapters/base.py
Abstract Base Sensor Adapter Contract.
Translates platform-specific sensor events into canonical SensorFrame objects.
"""

from abc import ABC, abstractmethod
from typing import Iterator, List, Any, Optional
from core.interfaces.canonical import SensorFrame


class BaseSensorAdapter(ABC):
    """Abstract Base Class for all Hardware / Device Sensor Adapters."""

    @abstractmethod
    def stream_sensor_frames(self) -> Iterator[SensorFrame]:
        """Yields continuous stream of normalized canonical SensorFrame instances."""
        pass

    @abstractmethod
    def get_adapter_metadata(self) -> dict:
        """Returns metadata about the hardware source, sampling capability, and driver version."""
        pass
