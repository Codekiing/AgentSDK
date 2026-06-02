"""
Self-contained chat template and tool parsers inlined from rllm.
Provides QwenChatTemplateParser, QwenToolParser, and helper functions
for converting messages to tokens and masks.
"""

import json
from typing import Any

from rllm_train.base import ToolCall


class ChatTemplateParser:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.assistant_token = ""

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    @classmethod
    def get_parser(cls, tokenizer, disable_thinking=False) -> "ChatTemplateParser":
        if isinstance(tokenizer.name_or_path, str):
            model_name = tokenizer.name_or_path.lower()
            tokenizer_cls = tokenizer.__class__.__name__.lower()
            if "qwen" in model_name or "qwen" in tokenizer_cls:
                return QwenChatTemplateParser(tokenizer, disable_thinking=disable_thinking)
        return ChatTemplateParser(tokenizer)


class QwenChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer, disable_thinking=True):
        super().__init__(tokenizer)
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.eot_token = "<|im_end|>\n"
        self.system_token = "<|im_start|>system\n"
        self.user_token = "<|im_start|>user\n"
        self.assistant_token = "<|im_start|>assistant\n"
        if disable_thinking:
            self.assistant_token += "<think>\\n\\n</think>\\n\\n"
        self.generation_prompt = self.assistant_token
        self.tool_start_token = "\n<tool_call>\n"
        self.tool_end_token = "\n</tool_call>"
        self.tool_response_start_token = "<tool_response>\n"
        self.tool_response_end_token = "\n</tool_response>"

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        result = ""
        if is_first_msg and messages[0]["role"] != "system":
            result += self.system_token + "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." + self.eot_token
        for message in messages:
            role = message["role"]
            if role == "system":
                result += self.system_token + message["content"] + self.eot_token
            elif role == "user":
                result += self.user_token + message["content"] + self.eot_token
            elif role == "assistant":
                result += self.assistant_token + message["content"] + self.eot_token
            elif role == "tool":
                result += self.user_token + self.tool_response_start_token + message["content"] + self.tool_response_end_token + self.eot_token
        if add_generation_prompt:
            result += self.generation_prompt
        return result


class QwenToolParser:
    def __init__(self):
        self.tool_call_begin = "<tool_call>"
        self.tool_call_end = "</tool_call>"

    def parse(self, model_response: str) -> list[ToolCall]:
        tool_calls_dicts, _ = self.parse_with_diagnostics(model_response)
        return [ToolCall(name=tc["name"], arguments=tc["arguments"]) for tc in tool_calls_dicts]

    def parse_with_diagnostics(self, model_response: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        text = str(model_response)
        tool_calls = self._parse_tool_calls(text)
        diagnostics = {
            "has_tool_tag": self.tool_call_begin in text,
            "parse_error_type": None,
            "malformed_tool_call": False,
            "empty_tool_call": False,
            "invalid_tool_call_count": 0,
            "recovered_missing_end_tag_count": 0,
        }
        if self.tool_call_begin not in text:
            return tool_calls, diagnostics
        recovered_count = sum(1 for call in tool_calls if call.get("_recovered_missing_end_tag"))
        diagnostics["recovered_missing_end_tag_count"] = recovered_count
        if self.tool_call_end not in text and not recovered_count:
            diagnostics["parse_error_type"] = "missing_end_tag"
            diagnostics["malformed_tool_call"] = True
        invalid_count = sum(1 for call in tool_calls if call.get("_invalid"))
        diagnostics["invalid_tool_call_count"] = invalid_count
        valid_calls = [
            {k: v for k, v in call.items() if not k.startswith("_")}
            for call in tool_calls
            if not call.get("_invalid")
        ]
        if invalid_count and diagnostics["parse_error_type"] is None:
            diagnostics["parse_error_type"] = "invalid_tool_call"
            diagnostics["malformed_tool_call"] = True
        if not valid_calls:
            diagnostics["empty_tool_call"] = True
        return valid_calls, diagnostics

    def _parse_tool_calls(self, text: str) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        if self.tool_call_begin not in text:
            return tool_calls
        while self.tool_call_begin in text:
            start = text.find(self.tool_call_begin) + len(self.tool_call_begin)
            end = text.find(self.tool_call_end)
            if end == -1:
                json_content = text[start:].strip()
                try:
                    call_data = json.loads(json_content)
                except json.JSONDecodeError:
                    tool_calls.append({"_invalid": True, "error": "missing_end_tag"})
                    break
                if self._is_valid_call_data(call_data):
                    tool_calls.append({
                        "name": call_data["name"],
                        "arguments": call_data["arguments"],
                        "_recovered_missing_end_tag": True,
                    })
                else:
                    tool_calls.append({"_invalid": True, "error": "missing_end_tag"})
                break
            json_content = text[start:end].strip()
            try:
                call_data = json.loads(json_content)
            except json.JSONDecodeError:
                tool_calls.append({"_invalid": True, "error": "invalid_json"})
                text = text[end + len(self.tool_call_end):]
                continue
            if not self._is_valid_call_data(call_data):
                tool_calls.append({"_invalid": True, "error": self._call_data_error(call_data)})
            else:
                tool_calls.append({"name": call_data["name"], "arguments": call_data["arguments"]})
            text = text[end + len(self.tool_call_end):]
        return tool_calls

    def _is_valid_call_data(self, call_data):
        return (
            isinstance(call_data, dict)
            and "name" in call_data
            and "arguments" in call_data
            and isinstance(call_data["arguments"], dict)
        )

    def _call_data_error(self, call_data):
        if not isinstance(call_data, dict):
            return "not_object"
        if "name" not in call_data:
            return "missing_name"
        if "arguments" not in call_data:
            return "missing_arguments"
        if not isinstance(call_data["arguments"], dict):
            return "arguments_not_object"
        return "invalid_tool_call"

    def get_tool_prompt(self, tools_schema: str) -> str:
        return f"""
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_schema}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call><|im_end|>
"""


def get_recent_assistant_user_messages(chat_completions_messages):
    env_messages = []
    assistant_message = None
    seen_assistant_message = False
    for message in reversed(chat_completions_messages):
        role = message.get("role", None)
        if role == "assistant":
            if assistant_message:
                break
            seen_assistant_message = True
            assistant_message = message
        elif role in ["user", "tool"] and not seen_assistant_message:
            env_messages.append(message)
    env_messages = list(reversed(env_messages))
    return assistant_message, env_messages


def convert_messages_to_tokens_and_masks(messages, tokenizer, parser, contains_first_msg=False, contains_generation_msg=False):
    all_msg_tokens = []
    all_msg_masks = []

    def _convert(msg, first_msg=False, generation_msg=False):
        msg_text = parser.parse([msg], add_generation_prompt=generation_msg, is_first_msg=first_msg)
        if msg["role"] == "assistant" and msg_text.startswith(parser.assistant_token):
            msg_text = msg_text.replace(parser.assistant_token, "", 1)
        msg_tokens = tokenizer.encode(msg_text, add_special_tokens=False)
        mask_value = 1 if msg["role"] == "assistant" else 0
        return msg_tokens, [mask_value] * len(msg_tokens)

    for i, msg in enumerate(messages):
        tokens, mask = _convert(
            msg,
            first_msg=(contains_first_msg and i == 0),
            generation_msg=(contains_generation_msg and i == len(messages) - 1),
        )
        all_msg_tokens.extend(tokens)
        all_msg_masks.extend(mask)

    return all_msg_tokens, all_msg_masks
