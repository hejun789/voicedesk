import json
import os
from voicedesk.llm import Message, ToolCall
from voicedesk.registry import TOOL_SCHEMAS


def _to_message(choice) -> Message:
    m = choice.message
    calls = []
    for tc in (m.tool_calls or []):
        calls.append(ToolCall(
            id=tc.id,
            name=tc.function.name,
            arguments=json.loads(tc.function.arguments or "{}"),
        ))
    return Message(content=m.content, tool_calls=calls)


class GroqLLM:
    def __init__(self, model: str = "llama-3.3-70b-versatile", api_key: str | None = None):
        from groq import Groq  # imported lazily so tests don't need the package
        self.model = model
        self.client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])

    def complete(self, messages: list[dict], tools: list[dict]) -> Message:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        return _to_message(resp.choices[0])
