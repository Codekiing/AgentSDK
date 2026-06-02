"""
Self-contained ToolAgent inlined from rllm.
"""

import copy
import json
import logging
import uuid
from typing import Any

from rllm_train.base import Action, BaseAgent, Step, ToolCall, Trajectory
from rllm_train.parsers import QwenToolParser

logger = logging.getLogger(__name__)


class ToolAgent(BaseAgent):
    def __init__(self, system_prompt="", parser_name="qwen", tool_map=None):
        self.system_prompt = system_prompt
        self.tool_parser = QwenToolParser()

        if tool_map is not None:
            self._tool_instances = {}
            for name, tool_cls in tool_map.items():
                self._tool_instances[name] = tool_cls()
            tools_json = [inst.json for inst in self._tool_instances.values()]
        else:
            self._tool_instances = {}
            tools_json = []

        self.tools_prompt = self.tool_parser.get_tool_prompt(json.dumps(tools_json, indent=2))
        self._trajectory = Trajectory()
        self.messages: list[dict[str, Any]] = []
        self.current_observation = None
        self.reset()

    def _format_observation_as_messages(self, obs: Any) -> list[dict]:
        messages = []
        if isinstance(obs, dict):
            if "question" in obs:
                messages.append({"role": "user", "content": obs["question"]})
            elif "tool_outputs" in obs:
                for tool_call_id, tool_output_str in obs["tool_outputs"].items():
                    messages.append({
                        "role": "user",
                        "content": f"<tool_response>\n{tool_output_str}\n</tool_response>",
                    })
        elif isinstance(obs, str):
            messages.append({"role": "user", "content": obs})
        elif obs:
            messages.append({"role": "user", "content": str(obs)})
        return messages

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        obs_messages = self._format_observation_as_messages(observation)
        self.messages.extend(obs_messages)
        self.current_observation = observation

    def update_from_model(self, response: str, **kwargs) -> Action:
        tool_calls_dict = []
        assistant_content = response
        parse_error = None
        diagnostics = {}
        try:
            if hasattr(self.tool_parser, "parse_with_diagnostics"):
                tool_calls_dicts, diagnostics = self.tool_parser.parse_with_diagnostics(response)
                tool_calls = [ToolCall(name=tc["name"], arguments=tc["arguments"]) for tc in tool_calls_dicts]
            else:
                tool_calls = self.tool_parser.parse(response)
            tool_calls_dict = [
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": tool_call.to_dict(),
                    "parsed_tool_call": True,
                    "synthetic_finish": False,
                    "parse_error_type": diagnostics.get("parse_error_type"),
                    "malformed_tool_call": bool(diagnostics.get("malformed_tool_call")),
                    "empty_tool_call": bool(diagnostics.get("empty_tool_call")),
                    "invalid_tool_call_count": int(diagnostics.get("invalid_tool_call_count", 0) or 0),
                    "recovered_missing_end_tag_count": int(diagnostics.get("recovered_missing_end_tag_count", 0) or 0),
                }
                for tool_call in tool_calls
            ]
            parse_error = diagnostics.get("parse_error_type")
        except Exception as e:
            logger.error(f"Failed to parse tool calls: {e}")
            parse_error = str(e)
            diagnostics = {"parse_error_type": parse_error, "malformed_tool_call": True}
            tool_calls_dict = []

        assistant_message = {"role": "assistant", "content": assistant_content}
        if tool_calls_dict:
            for call in tool_calls_dict:
                if isinstance(call.get("function", {}).get("arguments"), dict):
                    call["function"]["arguments"] = json.dumps(call["function"]["arguments"])
        else:
            tool_calls_dict = [
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "arguments": {"response": assistant_content},
                    },
                    "parsed_tool_call": False,
                    "synthetic_finish": True,
                    "parse_error_type": diagnostics.get("parse_error_type"),
                    "malformed_tool_call": bool(diagnostics.get("malformed_tool_call")),
                    "empty_tool_call": bool(diagnostics.get("empty_tool_call")),
                    "invalid_tool_call_count": int(diagnostics.get("invalid_tool_call_count", 0) or 0),
                    "recovered_missing_end_tag_count": int(diagnostics.get("recovered_missing_end_tag_count", 0) or 0),
                }
            ]

        self.messages.append(assistant_message)
        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            action=tool_calls_dict,
            model_response=response,
            observation=self.current_observation,
            info={
                "parsed_tool_call": any(call.get("parsed_tool_call") for call in tool_calls_dict),
                "synthetic_finish": any(call.get("synthetic_finish") for call in tool_calls_dict),
                "parse_error": parse_error,
                "parse_error_type": parse_error,
                "malformed_tool_call": bool(diagnostics.get("malformed_tool_call")),
                "empty_tool_call": bool(diagnostics.get("empty_tool_call")),
                "has_tool_tag": bool(diagnostics.get("has_tool_tag")),
                "invalid_tool_call_count": int(diagnostics.get("invalid_tool_call_count", 0) or 0),
                "recovered_missing_end_tag_count": int(diagnostics.get("recovered_missing_end_tag_count", 0) or 0),
            },
        )
        self._trajectory.steps.append(new_step)
        return Action(action=tool_calls_dict)

    def reset(self):
        self._trajectory = Trajectory()
        initial_prompt = (self.system_prompt or "") + (self.tools_prompt or "")
        self.messages = []
        if initial_prompt.strip():
            self.messages.append({
                "role": "user",
                "content": initial_prompt,
            })

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory
