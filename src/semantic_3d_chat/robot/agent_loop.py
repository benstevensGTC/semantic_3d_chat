"""Local constrained-JSON action loop for the embodied-camera precursor."""

from __future__ import annotations

import argparse
import json
import sys

from semantic_3d_chat.config import load_config
from semantic_3d_chat.robot.simulator import EmbodiedCameraSimulator
from semantic_3d_chat.robot.tools import RobotToolController, tool_schemas


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/default.yaml")
    result.add_argument("--scene", default="scene_000001")
    result.add_argument("--seed", type=int)
    result.add_argument("--command", help="Execute one constrained JSON tool envelope and exit")
    result.add_argument("--schemas", action="store_true", help="Print JSON tool schemas and exit")
    return result


def _write(payload: object) -> None:
    print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)


def main() -> None:
    args = parser().parse_args()
    config = load_config(args.config)
    if args.schemas:
        _write({"tools": tool_schemas(config)})
        return
    simulator = EmbodiedCameraSimulator(config, args.scene, seed=args.seed)
    controller = RobotToolController(simulator)
    _write(simulator.get_robot_state())
    if args.command is not None:
        try:
            payload = json.loads(args.command)
        except json.JSONDecodeError:
            _write(simulator.protocol_error("E_JSON"))
            return
        _write(controller.dispatch(payload))
        return
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            _write(simulator.protocol_error("E_JSON"))
            continue
        _write(controller.dispatch(payload))


# Console-script compatibility.
app = main


if __name__ == "__main__":
    main()
