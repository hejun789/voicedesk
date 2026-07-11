import json
import pytest
from voicedesk.evals.runner import (
    load_scenarios, tools_called_from, tool_calls_from, fresh_db, seed_db,
    all_appointments,
)


def test_tools_called_from_extracts_names_in_order():
    messages = [
        {"role": "system", "content": "x"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function",
             "function": {"name": "find_slots", "arguments": "{}"}},
            {"id": "2", "type": "function",
             "function": {"name": "book", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "1", "content": "{}"},
        {"role": "assistant", "content": "done"},
    ]
    assert tools_called_from(messages) == ["find_slots", "book"]


def test_tools_called_from_empty_when_no_tool_calls():
    assert tools_called_from([{"role": "assistant", "content": "hello"}]) == []


def test_tool_calls_from_returns_names_and_parsed_arguments_in_order():
    messages = [
        {"role": "system", "content": "x"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function",
             "function": {"name": "find_slots", "arguments": '{"date": "2026-07-13"}'}},
            {"id": "2", "type": "function",
             "function": {"name": "book", "arguments": '{"patient_name": "Jane Doe"}'}},
        ]},
        {"role": "tool", "tool_call_id": "1", "content": "{}"},
        {"role": "assistant", "content": "done"},
    ]
    assert tool_calls_from(messages) == [
        {"name": "find_slots", "arguments": {"date": "2026-07-13"}},
        {"name": "book", "arguments": {"patient_name": "Jane Doe"}},
    ]


def test_tool_calls_from_empty_when_no_tool_calls():
    assert tool_calls_from([{"role": "assistant", "content": "hello"}]) == []


def test_tool_calls_from_preserves_malformed_arguments_as_raw_string():
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function",
             "function": {"name": "book", "arguments": "{not json"}},
        ]},
    ]
    assert tool_calls_from(messages) == [
        {"name": "book", "arguments": {}, "arguments_raw": "{not json"},
    ]


def test_fresh_db_is_isolated():
    a, b = fresh_db(), fresh_db()
    seed_db(a, [{"patient_name": "Jane Doe", "phone": "5551234",
                 "slot_iso": "2026-07-13T09:00", "reason": "cleaning"}])
    assert len(all_appointments(a)) == 1
    assert all_appointments(b) == []  # separate DB, unaffected


def test_seed_db_then_all_appointments():
    conn = fresh_db()
    seed_db(conn, [{"patient_name": "Jane Doe", "phone": "5551234",
                    "slot_iso": "2026-07-13T09:00", "reason": "cleaning"}])
    appts = all_appointments(conn)
    assert appts == [{"patient_name": "Jane Doe", "phone": "5551234",
                      "slot_iso": "2026-07-13T09:00", "reason": "cleaning",
                      "status": "booked"}]


def test_seed_db_raises_on_impossible_slot():
    conn = fresh_db()
    with pytest.raises(ValueError):
        seed_db(conn, [{"patient_name": "X", "phone": "1",
                        "slot_iso": "2026-07-11T09:00",  # a Saturday
                        "reason": "cleaning"}])


def test_load_scenarios(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"id": "a", "turns": ["hi"]}]), encoding="utf-8")
    assert load_scenarios(str(p))[0]["id"] == "a"
