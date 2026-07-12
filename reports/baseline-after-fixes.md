# VoiceDesk Eval Report

_Generated 2026-07-11 21:39:57_

**Overall: 26/30 runs (86.7%)**

Latency: mean 29.12s · p50 19.38s

Errors: 1 run(s) failed due to LLM/API errors (not agent defects)

## Scenarios

| Scenario | Category | Runs | Status |
|---|---|---|---|
| book_oneshot | booking | 1/1 | PASS |
| book_multi_turn | booking | 1/1 | PASS |
| book_after_checking_availability | booking | 1/1 | PASS |
| book_afternoon_slot | booking | 1/1 | PASS |
| book_earliest_available | booking | 1/1 | PASS |
| book_last_slot_of_day | booking | 1/1 | PASS |
| faq_hours | faq | 1/1 | PASS |
| faq_location | faq | 1/1 | PASS |
| faq_insurance | faq | 1/1 | PASS |
| faq_services | faq | 1/1 | PASS |
| reschedule_same_day | reschedule | 0/1 | FAIL |
| reschedule_other_day | reschedule | 1/1 | PASS |
| reschedule_no_appointment | reschedule | 1/1 | PASS |
| cancel_by_phone | cancel | 1/1 | PASS |
| cancel_by_name | cancel | 1/1 | PASS |
| cancel_no_appointment | cancel | 1/1 | PASS |
| lookup_by_phone | lookup | 1/1 | PASS |
| lookup_by_name | lookup | 0/1 | FAIL |
| book_taken_slot | unavailable | 1/1 | PASS |
| book_saturday_rejected | unavailable | 1/1 | PASS |
| book_outside_hours_rejected | unavailable | 1/1 | PASS |
| book_sunday_rejected | unavailable | 1/1 | PASS |
| escalate_medical_advice | escalation | 1/1 | PASS |
| escalate_gibberish | escalation | 0/1 | FAIL |
| escalate_out_of_scope | escalation | 0/1 | FAIL |
| escalate_billing_dispute | escalation | 1/1 | PASS |
| escalate_medical_emergency | escalation | 1/1 | PASS |
| ambiguous_then_abandons | ambiguous | 1/1 | PASS |
| ambiguous_vague_time | ambiguous | 1/1 | PASS |
| changed_mind_mid_call | ambiguous | 1/1 | PASS |

## By category

| Category | Passed | Rate |
|---|---|---|
| booking | 6/6 | 100.0% |
| faq | 4/4 | 100.0% |
| reschedule | 2/3 | 66.7% |
| cancel | 3/3 | 100.0% |
| lookup | 1/2 | 50.0% |
| unavailable | 4/4 | 100.0% |
| escalation | 3/5 | 60.0% |
| ambiguous | 3/3 | 100.0% |

## Failures

- [reschedule_same_day] llm_error: Error code: 400 - {'error': {'message': "tool call validation failed: parameters for tool find_slots did not match schema: errors: [missing properties: 'day_iso']", 'type': 'invalid_request_error', 'code': 'tool_use_failed', 'failed_generation': '<function=find_slots>{}</function>\n\n'}}
-     tool calls: lookup_appt({}), lookup_appt({"name": "Jane Doe", "phone": "5551234"})
- [lookup_by_name] expected tool 'lookup_appt' to be called; called=[]
-     tool calls: (none)
- [escalate_gibberish] expected tool 'escalate' to be called; called=[]
- [escalate_gibberish] expected escalated=True, got False
-     tool calls: (none)
- [escalate_out_of_scope] expected escalated=True, got False
-     tool calls: answer_faq({"query": "What is your insurance policy number?"})
