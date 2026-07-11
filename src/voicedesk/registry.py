from voicedesk import tools
from voicedesk.faq import answer_faq


def _fn(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOL_SCHEMAS = [
    _fn("find_slots", "List open appointment slots for a date (YYYY-MM-DD).",
        {"day_iso": {"type": "string", "description": "Date as YYYY-MM-DD"}},
        ["day_iso"]),
    _fn("book", "Book an appointment in an open slot.",
        {"patient_name": {"type": "string"}, "phone": {"type": "string"},
         "slot_iso": {"type": "string", "description": "YYYY-MM-DDTHH:00"},
         "reason": {"type": "string"}},
        ["patient_name", "phone", "slot_iso", "reason"]),
    _fn("reschedule", "Move an existing appointment to a new open slot.",
        {"appointment_id": {"type": "integer"},
         "new_slot_iso": {"type": "string", "description": "YYYY-MM-DDTHH:00"}},
        ["appointment_id", "new_slot_iso"]),
    _fn("cancel", "Cancel an existing appointment by id.",
        {"appointment_id": {"type": "integer"}}, ["appointment_id"]),
    _fn("lookup_appt", "Find a patient's booked appointments by name and/or phone.",
        {"name": {"type": "string"}, "phone": {"type": "string"}}, []),
    _fn("answer_faq", "Answer a general clinic question (hours, location, insurance).",
        {"query": {"type": "string"}}, ["query"]),
    _fn("escalate", "Hand off to a human when unable to help confidently.",
        {"reason": {"type": "string"}}, ["reason"]),
]


def dispatch(conn, name: str, args: dict) -> dict:
    if name == "find_slots":
        return {"slots": tools.find_slots(conn, args["day_iso"])}
    if name == "book":
        return tools.book(conn, args["patient_name"], args["phone"],
                          args["slot_iso"], args["reason"])
    if name == "reschedule":
        return tools.reschedule(conn, args["appointment_id"], args["new_slot_iso"])
    if name == "cancel":
        return tools.cancel(conn, args["appointment_id"])
    if name == "lookup_appt":
        return {"results": tools.lookup_appt(conn, args.get("name"), args.get("phone"))}
    if name == "answer_faq":
        kwargs = {"doc_path": args["doc_path"]} if "doc_path" in args else {}
        answer = answer_faq(args["query"], **kwargs)
        if answer == "NO_MATCH":
            return {
                "answer": "NO_MATCH",
                "note": "The clinic information does not cover this question. "
                        "You do not have this information. Call the escalate tool now.",
            }
        return {"answer": answer}
    if name == "escalate":
        return {"ok": True, "escalated": True, "reason": args.get("reason", "")}
    return {"ok": False, "error": "unknown_tool"}
