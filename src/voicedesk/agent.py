import json
from voicedesk.llm import LLMClient, Message
from voicedesk.registry import TOOL_SCHEMAS, dispatch

DEFAULT_SYSTEM_PROMPT = (
    "You are the phone receptionist for BrightSmile Dental. "
    "Use the provided tools to find slots and to book, reschedule, cancel, or "
    "look up appointments, and to answer general questions. "
    "Always confirm the patient's name, phone, and desired time before booking. "
    "If a tool reports slot_unavailable, offer other open slots. "
    "If you cannot help confidently, or input is unclear or out of scope, call the "
    "escalate tool. Keep replies short and natural, as if speaking on a phone call."
)

MAX_ITERS = 5
_FALLBACK = (
    "I'm having trouble with that. Let me have a team member call you back."
)


class Agent:
    def __init__(self, conn, llm: LLMClient, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.conn = conn
        self.llm = llm
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]

    def respond(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        for _ in range(MAX_ITERS):
            msg: Message = self.llm.complete(self.messages, TOOL_SCHEMAS)
            if not msg.tool_calls:
                text = msg.content or _FALLBACK
                self.messages.append({"role": "assistant", "content": text})
                return text
            self.messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                result = dispatch(self.conn, tc.name, tc.arguments)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })
        self.messages.append({"role": "assistant", "content": _FALLBACK})
        return _FALLBACK
