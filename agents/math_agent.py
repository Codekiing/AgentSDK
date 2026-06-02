from typing import List, Any, Dict

from agentic_rl.runner.agent_engine_wrapper.base import Trajectory
from agentic_rl.runner.agent_engine_wrapper.base_engine_wrapper import BaseEngineWrapper


class MathAgent(BaseEngineWrapper):
    def initialize(self):
        pass

    def generate_agent_trajectories_async(self, tasks: List[dict]) -> List[Trajectory]:
        trajectories = []
        for task in tasks:
            traj = Trajectory(
                prompt=task.get("prompt", ""),
                response="",
                messages=task.get("messages", []),
                reward=0.0,
            )
            trajectories.append(traj)
        return trajectories
