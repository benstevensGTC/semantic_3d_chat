"""Continuous robot-state tokens for the embodied scene prefix."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class NumericRobotState:
    position_m: tuple[float, float, float]
    body_yaw_degrees: float
    camera_yaw_degrees: float
    pitch_degrees: float
    linear_velocity_xy_m: tuple[float, float]
    angular_velocity_degrees: float
    collision: bool
    last_movement_delta_m: tuple[float, float, float]
    scan_coverage: float
    stopped: bool


ROBOT_STATE_FEATURE_DIM = 18


def robot_state_vector(
    state: NumericRobotState,
    room_min_m: torch.Tensor,
    room_max_m: torch.Tensor,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Encode state numerically; no pose or observation is converted to prose."""

    room_min = torch.as_tensor(room_min_m, dtype=torch.float32, device=device)
    room_max = torch.as_tensor(room_max_m, dtype=torch.float32, device=device)
    if room_min.shape != (3,) or room_max.shape != (3,) or torch.any(room_max <= room_min):
        raise ValueError("room bounds must be finite three-vectors with max > min")
    position = torch.tensor(state.position_m, dtype=torch.float32, device=device)
    movement = torch.tensor(state.last_movement_delta_m, dtype=torch.float32, device=device)
    velocity = torch.tensor(state.linear_velocity_xy_m, dtype=torch.float32, device=device)
    normalized_position = 2.0 * (position - room_min) / (room_max - room_min) - 1.0
    normalized_movement = movement / torch.clamp(room_max - room_min, min=1e-6)
    body_yaw = math.radians(float(state.body_yaw_degrees))
    camera_yaw = math.radians(float(state.camera_yaw_degrees))
    pitch = math.radians(float(state.pitch_degrees))
    values = torch.cat(
        (
            normalized_position,
            torch.tensor(
                [
                    math.sin(body_yaw),
                    math.cos(body_yaw),
                    math.sin(camera_yaw),
                    math.cos(camera_yaw),
                    math.sin(pitch),
                    math.cos(pitch),
                ],
                dtype=torch.float32,
                device=device,
            ),
            velocity,
            torch.tensor(
                [
                    float(state.angular_velocity_degrees) / 180.0,
                    float(state.collision),
                ],
                dtype=torch.float32,
                device=device,
            ),
            normalized_movement,
            torch.tensor(
                [float(state.scan_coverage), float(state.stopped)],
                dtype=torch.float32,
                device=device,
            ),
        )
    )
    if values.shape != (ROBOT_STATE_FEATURE_DIM,) or not torch.isfinite(values).all():
        raise ValueError("Robot state produced an invalid numeric feature vector")
    return values


class RobotStateEncoder(nn.Module):
    """Trainable MLP that emits one or more continuous LM-space state tokens."""

    def __init__(self, output_dim: int, *, hidden_dim: int = 256, token_count: int = 4) -> None:
        super().__init__()
        if output_dim < 1 or hidden_dim < 1 or token_count < 1:
            raise ValueError("output_dim, hidden_dim, and token_count must be positive")
        self.output_dim = int(output_dim)
        self.token_count = int(token_count)
        self.network = nn.Sequential(
            nn.Linear(ROBOT_STATE_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.token_count * self.output_dim),
        )

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        if state_features.ndim == 1:
            state_features = state_features.unsqueeze(0)
        if state_features.ndim != 2 or state_features.shape[-1] != ROBOT_STATE_FEATURE_DIM:
            raise ValueError(
                f"state_features must have shape [B, {ROBOT_STATE_FEATURE_DIM}]"
            )
        tokens = self.network(state_features).reshape(
            state_features.shape[0], self.token_count, self.output_dim
        )
        if not torch.isfinite(tokens).all():
            raise RuntimeError("Robot-state tokens contain NaN or infinity")
        return tokens


def append_robot_state_tokens(scene_prefix: torch.Tensor, robot_tokens: torch.Tensor) -> torch.Tensor:
    """Place continuous robot-state tokens beside a continuous scene prefix."""

    if scene_prefix.ndim != 3 or robot_tokens.ndim != 3:
        raise ValueError("scene_prefix and robot_tokens must both be rank-three")
    if scene_prefix.shape[0] != robot_tokens.shape[0]:
        raise ValueError("scene and robot token batch sizes differ")
    if scene_prefix.shape[2] != robot_tokens.shape[2]:
        raise ValueError("scene and robot hidden dimensions differ")
    return torch.cat((scene_prefix, robot_tokens.to(scene_prefix)), dim=1)
