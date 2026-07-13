import json
from datetime import date
from voicedesk.llm import LLMClient, LLMError, Message
from voicedesk.registry import TOOL_SCHEMAS, dispatch


def build_system_prompt(today: date) -> str:
    today_str = today.strftime("%A, %d %B %Y")
    return (
        f"You are the phone receptionist for BrightSmile Dental. "
        f"Today is {today_str}. Resolve every relative date (\"Monday\", "
        f"\"next week\", \"tomorrow\") against today's date, and always pass "
        f"absolute YYYY-MM-DDTHH:00 values to the tools. Never book a date in "
        f"the past. "
        "Use the provided tools to find slots and to book, reschedule, cancel, or "
        "look up appointments, and to answer general questions. "
        "Before calling the book tool you MUST read the details back to the caller "
        "and get an explicit yes. Repeat the name, the phone number DIGIT BY DIGIT "
        "(for example \"five five five, one two three four\"), the date and time, "
        "and the reason. Speech-to-text frequently mis-hears digits, so the caller "
        "must confirm the number before anything is written. Do NOT call book on "
        "the same turn the caller first gives their details — ask for confirmation "
        "first, then book once they confirm. "
        "Never invent or guess patient details. If you do not yet know the "
        "caller's real name or phone number, ASK for it. Never pass placeholder "
        "values such as \"unknown\", \"N/A\", or the parameter names themselves "
        "to any tool. "
        "If a tool reports slot_unavailable, offer other open slots. "
        "After calling answer_faq, you MUST relay the retrieved answer to the caller "
        "in your reply. Never ignore it and change the subject. If answer_faq returns "
        "NO_MATCH, say you don't have that information and call escalate. "
        "To reschedule or cancel, you MUST first call lookup_appt and use the exact "
        "appointment_id it returns. NEVER invent or guess an appointment_id. "
        "Never say an appointment_id, database id, or any internal identifier out loud "
        "to the caller. Ids are for tool calls only. Confirm a booking by repeating "
        "the date, time and reason instead. "
        "Call escalate whenever the caller reports a medical problem, symptom, pain, "
        "injury or emergency; asks for medical advice; raises a billing dispute or "
        "refund; or asks for anything outside booking, rescheduling, cancelling, or "
        "clinic information. Do this even if you could compose a plausible reply — "
        "escalating is always the safe choice for these. "
        "If you cannot help confidently, or input is unclear or out of scope, call the "
        "escalate tool. "
        "If the caller's message is unintelligible, gibberish, empty of meaning, or you "
        "cannot determine what they want, call the escalate tool. Do not guess and do not "
        "reply with small talk. "
        "lookup_appt works with a name ALONE or a phone number ALONE — you do not need "
        "both. If the caller gives you either one, call lookup_appt with what you have "
        "instead of asking for more. "
        "If the caller changes the time or day of an appointment that already exists — "
        "including one you booked earlier in this same call — you MUST look it up and "
        "reschedule it. NEVER book a second appointment for the same person; they must "
        "end the call with exactly one appointment. "
        "If a tool reports already_booked_that_day, that patient already has an "
        "appointment that day. Do not book again — call lookup_appt and then reschedule. "
        "Keep replies short and natural, as if speaking on a phone call."
    )


MAX_ITERS = 5
_FALLBACK = (
    "I'm having trouble with that. Let me have a team member call you back."
)


class Agent:
    def __init__(self, conn, llm: LLMClient, system_prompt: str | None = None):
        self.conn = conn
        self.llm = llm
        if system_prompt is None:
            system_prompt = build_system_prompt(date.today())
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]

    def respond(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        for _ in range(MAX_ITERS):
            try:
                msg: Message = self.llm.complete(self.messages, TOOL_SCHEMAS)
            except LLMError:
                self.messages.append({"role": "assistant", "content": _FALLBACK})
                return _FALLBACK
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
                try:
                    result = dispatch(self.conn, tc.name, tc.arguments)
                except Exception as e:
                    result = {"ok": False, "error": "tool_error", "detail": str(e)}
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })
        self.messages.append({"role": "assistant", "content": _FALLBACK})
        return _FALLBACK
