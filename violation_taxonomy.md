# Violation taxonomy v1 — clip analysis and mapping to norms

Method: full-length timestamped contact sheets for all 42 clips, then native-resolution frame strips around each labeled second, then matched violation/control pairs. Rule texts are in `rules_kb.json` (verbatim, read on adilet).

Confidence scale: **A** = visible difference vs. a matched control, norm text fits. **B** = visible event fits a norm, but no clean control comparison. **C** = unresolved, label not explained by what I could see.

## Key finding: labels are about *state*, not motion

The cashier walking to the safe and leaving the room appears in both violation and no-violation clips. Matched pairs show what differs:

| Pair | Same in both | What differs | Evidence |
|---|---|---|---|
| X-28 (viol, 7 s) vs X-29 (ctrl) | works at laptop, goes to safe, leaves room | **laptop left unlocked** (work spreadsheet stays on) vs. locked Windows screen as she stands up | zoom on laptop, `z/lap28.png`, `z/lap29.png` |
| X-11 (viol, 18 s) vs X-23 (ctrl) | same sequence | laptop still lit with the work screen while she leaves; control shows the blue lock screen | `z/laptops.png` |
| X-26 (viol, 9 s) vs X-27 (ctrl) | same layout and route to the safe, then out | **banknote left on the desk** when she leaves (visible 14–18 s); control desk is empty | `z/x26.png`, `z/x27.png` |
| X-15 (viol, 15 s) | opens safe, leaves | **green cash bag left on desk**, she leaves the room | `z/v15.png` |
| X-40 (viol, 17 s) | — | **coat and patterned handbag placed on the cash desk**; she works with them there | `z/v40.png` |
| X-46 (viol, 11 s) | — | **cup and food on the desk**, she drinks at the workstation | `z/v46.png` |
| X-30 (viol, 13 s) | — | a man walks **behind the desk** and stands next to the cashier at the counting machine | `z/v30.png` |

Hypothesis that may unify several others: **cash left lying on the desk** (X-4, X-38, X-13, X-26, X-15) — see T2 below.

Rejected: "counted by hand instead of on the counting machine". Control X-3 shows the same manual handling as X-1 and X-47.

## Types

| # | Type name (shown to user) | What is visible | Norm | Link strength |
|---|---|---|---|---|
| T1 | Наличные оставлены на рабочем столе | banknotes / cash bag on the desk while cashier is away or while another client arrives | НБРК № 231, п. 20 ч. 2: «Все ранее принятые наличные деньги хранятся в специальных устройствах … закрывающихся на ключ» | **direct** |
| T2 | Личные вещи на рабочем месте | coat, bag at the cash desk | НБРК № 231, п. 12: «Хранение личных вещей … осуществляется вне кассового узла» | **direct** for belongings |
| T3 | Еда / напитки на рабочем месте | cup, food on the cash desk | same п. 12 | weak — food and drink are not named; frame as "personal items" |
| T4 | Посторонний в рабочей зоне кассира | non-cashier walks behind the desk to the cashier | НБРК № 14, п. 8 (isolation of cash node); НБРК № 231, п. 75 (by analogy, vault access) | **indirect** — neutral wording only |
| T5 | Рабочий компьютер не заблокирован при уходе | cashier leaves with the work screen on | АРРФР № 53, п. 71 пп. 9 (system auto-lock after 30 min) | **indirect / weak** — see caveat |
| T6 | Уход с рабочего места во время обслуживания клиента | cashier leaves while a client is being served | НБРК № 231, п. 22 | direct, **only if a client is in the room**. I found no clip with this situation. |

### Numbering fix
In the first draft T1 was "cashier leaves". That is not a violation by itself (see controls X-23, X-25, X-27, X-29, which all leave). It became a *context* of T1, T5 or T6. Leaving is a trigger for the state check, never the verdict.

## Per-clip table (22 violation clips)

| Clip | t | What I saw at/around the label | Type | Conf. |
|---|---|---|---|---|
| X-28 | 7 | stands up from laptop with spreadsheet still on screen, goes out | T5 | A |
| X-11 | 18 | laptop screen lit as she leaves, room lights switch off (brightness drop) | T5 | A |
| X-26 | 9 | turns away from desk; note stays on the desk until the end | T1 | A |
| X-15 | 15 | leaves room, green cash bag stays on desk | T1 | A |
| X-40 | 17 | coat + handbag on desk | T2 | A |
| X-46 | 11 | picks up cup, drinks, food on desk | T3 | A (visual) / weak norm |
| X-30 | 13 | man in dark jacket steps to the cashier's side of the desk | T4 | B |
| X-38 | 33 | previous client gone (≈16 s), two small banknote packs still on desk when new visitor enters at 31.5–33 s | T1 | B |
| X-4 | 57 | client steps out (≈55 s) with banknotes lying on the desk, returns at 57 s | T1 (or п. 20 ч. 1, counting without client) | B |
| X-13 | 47 | cash on desk (40–44 s) while she stands at the open safe; shuttles cash by hand desk → safe | T1 | B |
| X-44 | 16 | counts banknotes by hand alone at the desk; small object in left hand at 13 s | п. 42 (no controlling employee)? | C |
| X-17 | 7 | man in dark jacket carries a green bag in, puts it on the desk, she takes it | bag handover (п. 24–25)? | C |
| X-18 | 20 | second visitor (white shirt) enters while first is still at the desk, hands something over | second person at the desk? | C |
| X-22 | 14 | leaves after closing safe; laptop already on lock screen at 10 s; safe door open earlier while she sat | — | C |
| X-24 | 8 | goes to safe, leaves; tablet and cans stay on desk, safe closed | — | C |
| X-20 | 5 | almost identical to control X-21 (safe, cash, sits) | — | C |
| X-32 | 5 | workplace empty from 0 s; she enters through the door at ≈5 s | workplace unattended? | C |
| X-33 | 6 | same: empty workplace, she enters at 4.5–6 s | workplace unattended? | C |
| X-34 | 13 | visitor in white leaves the room, she reaches to the counting machine | — | C |
| X-1 | 22 | client at desk, she counts by hand, he reaches toward the money at 23.5 s | — | C |
| X-2 | 10 | client enters, puts a note on the desk, she later reaches across the desk with cash | — | C |
| X-47 | 68 | long clip, client and cashier exchange cash; moment unclear | — | C |

Counts: **A = 6, B = 4, C = 12**. Only the six A-clips are solid enough to call "understood". Detection and classification scores depend heavily on the C-clips, so the next step must be clarifying them, not building more machinery.

## What this means for the design

1. **Verdict = state check, not motion detection.** Candidate-event extractor (person leaves desk, new person enters, cash/bag visible on desk) must be followed by a state question: *is cash on the desk? is the laptop screen lit? are personal items on the desk?*
2. **Cash/bag on the desk is the best-supported, most reusable signal** (T1: clips X-26, X-15, and likely X-4, X-38, X-13) and has the strongest norm. Prioritize it.
3. **T5 (laptop lock) is detectable but legally weak**, and the laptop is only usable in 2 of 22 violation clips (others have dark screens). A small screen-state classifier is cheap, but report it as "screen left unlocked", never as a legal breach.
4. **Robustness to angle:** the discriminating objects (cash, bag, laptop screen, coat) are small. From a different angle, a fixed crop will fail; use open-vocabulary detection on the full frame, plus VLM verification.
5. **Timestamp convention:** labels mix "start of the action" (X-28: she stands up), "moment of consequence" (X-38: visitor enters while cash is on the desk) and "re-entry" (X-32/33). Our output should use the first frame where the *state* becomes violating.

## Open questions for the team / organizers

1. Which norm did the dataset authors intend for the unlocked laptop (T5)? The only open candidate I found is the 2026 information-security act; the 2018 act it replaces may be the reference.
2. Is "cash left on desk between clients" (T1) what X-4 and X-38 are about?
3. What are X-32 / X-33 (empty workplace at the start)? Is that a "workplace left unattended" violation, and what would the timecode be?
4. Does the final test include violation types that are not in training (FAQ 9)? Yes — so T-types should stay open-ended and descriptions neutral.
