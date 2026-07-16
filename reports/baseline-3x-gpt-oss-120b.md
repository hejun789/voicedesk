# VoiceDesk Eval Report

_Generated 2026-07-16 11:12:00_

**Model:** `openai/gpt-oss-120b`

**Overall: 76/90 runs (84.4%)**

Latency: mean 13.19s · p50 1.86s

## Scenarios

| Scenario | Category | Runs | Status |
|---|---|---|---|
| book_oneshot | booking | 3/3 | PASS |
| book_multi_turn | booking | 2/3 | FLAKY |
| book_after_checking_availability | booking | 3/3 | PASS |
| book_afternoon_slot | booking | 3/3 | PASS |
| book_earliest_available | booking | 2/3 | FLAKY |
| book_last_slot_of_day | booking | 3/3 | PASS |
| faq_hours | faq | 3/3 | PASS |
| faq_location | faq | 2/3 | FLAKY |
| faq_insurance | faq | 3/3 | PASS |
| faq_services | faq | 3/3 | PASS |
| reschedule_same_day | reschedule | 1/3 | FLAKY |
| reschedule_other_day | reschedule | 0/3 | FAIL |
| reschedule_no_appointment | reschedule | 2/3 | FLAKY |
| cancel_by_phone | cancel | 0/3 | FAIL |
| cancel_by_name | cancel | 2/3 | FLAKY |
| cancel_no_appointment | cancel | 3/3 | PASS |
| lookup_by_phone | lookup | 3/3 | PASS |
| lookup_by_name | lookup | 3/3 | PASS |
| book_taken_slot | unavailable | 3/3 | PASS |
| book_saturday_rejected | unavailable | 3/3 | PASS |
| book_outside_hours_rejected | unavailable | 3/3 | PASS |
| book_sunday_rejected | unavailable | 3/3 | PASS |
| escalate_medical_advice | escalation | 3/3 | PASS |
| escalate_gibberish | escalation | 3/3 | PASS |
| escalate_out_of_scope | escalation | 2/3 | FLAKY |
| escalate_billing_dispute | escalation | 3/3 | PASS |
| escalate_medical_emergency | escalation | 3/3 | PASS |
| ambiguous_then_abandons | ambiguous | 3/3 | PASS |
| ambiguous_vague_time | ambiguous | 3/3 | PASS |
| changed_mind_mid_call | ambiguous | 3/3 | PASS |

## By category

| Category | Passed | Rate |
|---|---|---|
| booking | 16/18 | 88.9% |
| faq | 11/12 | 91.7% |
| reschedule | 3/9 | 33.3% |
| cancel | 5/9 | 55.6% |
| lookup | 6/6 | 100.0% |
| unavailable | 12/12 | 100.0% |
| escalation | 14/15 | 93.3% |
| ambiguous | 9/9 | 100.0% |

## Failures

- [book_multi_turn] expected tool 'book' to be called; called=[]
- [book_multi_turn] no appointment matching {'patient_name': 'John Smith', 'slot_iso': '2026-07-13T10:00', 'status': 'booked'}; actual=[]
-     tool calls: (none)
- [book_earliest_available] expected tool 'book' to be called; called=['find_slots']
- [book_earliest_available] no appointment matching {'patient_name': 'Anna Kim', 'slot_iso': '2026-07-16T09:00', 'status': 'booked'}; actual=[]
-     tool calls: find_slots({"day_iso": "2026-07-16"})
- [faq_location] reply did not contain 'Market Street'; reply='BrightSmile Dental is located at\u202f200\u202fMarket\u202fStreet, Suite\u202f4, Springfield. Parking is free behind the building.'
-     tool calls: answer_faq({"query": "location"})
- [reschedule_same_day] expected tool 'reschedule' to be called; called=['find_slots', 'lookup_appt']
- [reschedule_same_day] no appointment matching {'patient_name': 'Jane Doe', 'slot_iso': '2026-07-13T11:00', 'status': 'booked'}; actual=[{'patient_name': 'Jane Doe', 'phone': '5551234', 'slot_iso': '2026-07-13T09:00', 'reason': 'cleaning', 'status': 'booked'}]
-     tool calls: lookup_appt({"name": "Jane Doe"}), find_slots({"day_iso": "2026-07-13"})
- [reschedule_same_day] expected tool 'reschedule' to be called; called=['find_slots', 'lookup_appt']
- [reschedule_same_day] no appointment matching {'patient_name': 'Jane Doe', 'slot_iso': '2026-07-13T11:00', 'status': 'booked'}; actual=[{'patient_name': 'Jane Doe', 'phone': '5551234', 'slot_iso': '2026-07-13T09:00', 'reason': 'cleaning', 'status': 'booked'}]
-     tool calls: lookup_appt({"name": "Jane Doe"}), find_slots({"day_iso": "2026-07-13"})
- [reschedule_other_day] expected tool 'reschedule' to be called; called=['find_slots', 'lookup_appt']
- [reschedule_other_day] no appointment matching {'patient_name': 'John Smith', 'slot_iso': '2026-07-14T09:00', 'status': 'booked'}; actual=[{'patient_name': 'John Smith', 'phone': '5559876', 'slot_iso': '2026-07-13T10:00', 'reason': 'filling', 'status': 'booked'}]
-     tool calls: lookup_appt({"name": "John Smith"}), find_slots({"day_iso": "2026-07-14"})
- [reschedule_other_day] expected tool 'reschedule' to be called; called=['find_slots', 'lookup_appt']
- [reschedule_other_day] no appointment matching {'patient_name': 'John Smith', 'slot_iso': '2026-07-14T09:00', 'status': 'booked'}; actual=[{'patient_name': 'John Smith', 'phone': '5559876', 'slot_iso': '2026-07-13T10:00', 'reason': 'filling', 'status': 'booked'}]
-     tool calls: lookup_appt({"name": "John Smith"}), find_slots({"day_iso": "2026-07-14"})
- [reschedule_other_day] expected tool 'lookup_appt' to be called; called=[]
- [reschedule_other_day] expected tool 'reschedule' to be called; called=[]
- [reschedule_other_day] no appointment matching {'patient_name': 'John Smith', 'slot_iso': '2026-07-14T09:00', 'status': 'booked'}; actual=[{'patient_name': 'John Smith', 'phone': '5559876', 'slot_iso': '2026-07-13T10:00', 'reason': 'filling', 'status': 'booked'}]
-     tool calls: (none)
- [reschedule_no_appointment] expected tool 'lookup_appt' to be called; called=[]
-     tool calls: (none)
- [cancel_by_phone] expected tool 'cancel' to be called; called=['lookup_appt']
- [cancel_by_phone] no appointment matching {'patient_name': 'Jane Doe', 'slot_iso': '2026-07-13T09:00', 'status': 'cancelled'}; actual=[{'patient_name': 'Jane Doe', 'phone': '5551234', 'slot_iso': '2026-07-13T09:00', 'reason': 'cleaning', 'status': 'booked'}]
-     tool calls: lookup_appt({"name": "Jane Doe", "phone": "5551234"})
- [cancel_by_phone] expected tool 'lookup_appt' to be called; called=[]
- [cancel_by_phone] expected tool 'cancel' to be called; called=[]
- [cancel_by_phone] no appointment matching {'patient_name': 'Jane Doe', 'slot_iso': '2026-07-13T09:00', 'status': 'cancelled'}; actual=[{'patient_name': 'Jane Doe', 'phone': '5551234', 'slot_iso': '2026-07-13T09:00', 'reason': 'cleaning', 'status': 'booked'}]
-     tool calls: (none)
- [cancel_by_phone] expected tool 'cancel' to be called; called=['lookup_appt']
- [cancel_by_phone] no appointment matching {'patient_name': 'Jane Doe', 'slot_iso': '2026-07-13T09:00', 'status': 'cancelled'}; actual=[{'patient_name': 'Jane Doe', 'phone': '5551234', 'slot_iso': '2026-07-13T09:00', 'reason': 'cleaning', 'status': 'booked'}]
-     tool calls: lookup_appt({"name": "Jane Doe", "phone": "5551234"})
- [cancel_by_name] expected tool 'cancel' to be called; called=['lookup_appt']
- [cancel_by_name] no appointment matching {'patient_name': 'Mary Lee', 'slot_iso': '2026-07-14T14:00', 'status': 'cancelled'}; actual=[{'patient_name': 'Mary Lee', 'phone': '5552222', 'slot_iso': '2026-07-14T14:00', 'reason': 'checkup', 'status': 'booked'}]
-     tool calls: lookup_appt({"name": "Mary Lee"})
- [escalate_out_of_scope] expected escalated=True, got False
-     tool calls: (none)
