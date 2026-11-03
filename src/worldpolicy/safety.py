"""Fail-closed real-robot gate; no motor transport exists in this package."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotGate:
    enabled: bool = False
    identity_verified: bool = False
    calibrated: bool = False
    supervision_confirmed: bool = False
    emergency_stop_verified: bool = False

    def assert_motion_allowed(self) -> None:
        if not all((self.enabled, self.identity_verified, self.calibrated,
                    self.supervision_confirmed, self.emergency_stop_verified)):
            raise PermissionError("Real robot motion gate is closed")
