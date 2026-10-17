# Per-video rule mapping

Rule and confidence come from my frame-by-frame analysis (A = visible difference vs a matched control and the norm fits, B = event fits the norm, no clean control, C = label unexplained). 'Pipeline' is the eval output at threshold 0.5 (`runs/eval`). 'Type match' compares the type I assigned with what fired.

## Violation clips

| Clip | Label | What happens | Rule (my analysis) | Conf | Pipeline @0.5 | Verdict | Type match |
|---|---|---|---|---|---|---|---|
| X-1 | 22 s | Client sits across; cashier counts banknotes by hand; client reaches toward the money at ~23.5 s. Manual counting is normal (control X-3 does it too). | — | C | — | MISSED | n/a (unexplained) |
| X-2 | 10 s | Client enters, puts a note on the desk at ~8.5-10 s; later cashier reaches across the desk with cash. | — | C | — | MISSED | n/a (unexplained) |
| X-4 | 57 s | Client leaves at ~55 s with banknotes lying on the desk, returns at 57 s; cashier stays seated. | T1: п. 20 (часть 2) (НБРК №231); п. 20 (часть 1) (НБРК №231) | B | — | MISSED | no |
| X-11 | 18 s | Cashier walks away from the desk; laptop still shows the work screen. Room lights switch off. | T5: п. 71 пп. 9 (АРРФР №53) | A | T7@17s(0.50) | caught | no |
| X-13 | 47 s | Cash lies on the desk (40-44 s) while cashier stands at the open safe, moving cash by hand between desk and safe. | T1: п. 20 (часть 2) (НБРК №231) | B | T1@22s(0.92), T7@58s(0.50) | caught | yes |
| X-15 | 15 s | Green cash bag stays on the desk while the cashier walks out of the room. | T1: п. 20 (часть 2) (НБРК №231) | A | T1@0s(0.95), T9@1s(0.60), T7@16s(0.50) | caught | yes |
| X-17 | 7 s | Man in dark jacket brings a green bag in, puts it on the desk, cashier takes it, he leaves. Candidate rules (unverified): collector crew/kit, bag checks. | п. 98 (НБРК №231); п. 103 (НБРК №231); пп. 24–25 (НБРК №231) | C | — | MISSED | n/a (unexplained) |
| X-18 | 20 s | A second visitor (white shirt) enters while the first is still at the desk and hands something over. | — | C | — | MISSED | n/a (unexplained) |
| X-20 | 5 s | Cashier at the safe, then sits. Almost identical to control X-21. | — | C | — | MISSED | n/a (unexplained) |
| X-22 | 14 s | Cashier closes the safe and leaves; laptop already on the lock screen at 10 s; safe door was open while she sat (6.5-9.5 s). | — | C | T7@17s(0.50) | caught | n/a (unexplained) |
| X-24 | 8 s | Cashier goes to the safe and leaves; tablet and cans stay on the desk; safe closed. | — | C | T3@0s(0.65) | caught | n/a (unexplained) |
| X-26 | 9 s | Cashier leaves; a banknote stays on the desk until the end. Control X-27 has an empty desk. | T1: п. 20 (часть 2) (НБРК №231) | A | T1@10s(0.91) | caught | yes |
| X-28 | 7 s | Cashier stands up and leaves with the laptop still on the work screen. Control X-29 locks the screen first. | T5: п. 71 пп. 9 (АРРФР №53) | A | — | MISSED | no |
| X-30 | 13 s | Man in dark jacket walks behind the desk and stands next to the cashier at the counting machine. | T4: п. 8 (НБРК №14); п. 75 (НБРК №231) | B | — | MISSED | no |
| X-32 | 5 s | Workplace empty from 0 s; cashier enters through the door at ~5 s. | — | C | T7@0s(0.50) | caught | n/a (unexplained) |
| X-33 | 6 s | Workplace empty; cashier enters at 4.5-6 s; a man walks by behind the glass at 10.5 s. | — | C | — | MISSED | n/a (unexplained) |
| X-34 | 13 s | Visitor in white leaves the room, cashier reaches toward the counting machine. | — | C | — | MISSED | n/a (unexplained) |
| X-38 | 33 s | First client gone by ~16 s; two small banknote packs still on the desk when a new visitor enters at 31.5-33 s. | T1: п. 20 (часть 2) (НБРК №231) | B | — | MISSED | no |
| X-40 | 17 s | Cashier puts a beige coat and a patterned handbag on the cash desk and works beside them. | T2: п. 12 (НБРК №231) | A | T7@3s(0.50) | caught | no |
| X-44 | 16 s | Cashier counts banknotes by hand alone at the desk; small object in her left hand at 13 s. Possible: counting without a controlling employee. | п. 42 (НБРК №231) | C | — | MISSED | n/a (unexplained) |
| X-46 | 11 s | Cashier picks up a cup and drinks at the desk; food on the desk. | T3: п. 12 (НБРК №231) | A | — | MISSED | no |
| X-47 | 68 s | Long clip; client and cashier exchange cash; moment unclear. | — | C | — | MISSED | n/a (unexplained) |

Verdict caught 8 of 22; fired the type I assigned on 3 of 10 clips that have one.

## Control clips (no violation)

| Clip | Note | Pipeline @0.5 | Result |
|---|---|---|---|
| X-3 | Man enters, sits, hands money, cashier counts by hand; leaves. Normal transaction. | — | ok |
| X-5 | not inspected (byte-identical file to X-48) | — | ok |
| X-8 | not inspected | T3@13s(0.57) | FALSE ALARM |
| X-10 | not inspected | — | ok |
| X-12 | not inspected | T2@19s(0.55), T7@36s(0.50) | FALSE ALARM |
| X-14 | not inspected | T1@0s(0.75), T1@35s(0.95) | FALSE ALARM |
| X-16 | not inspected | — | ok |
| X-19 | not inspected | — | ok |
| X-21 | Cashier takes cash from the safe and sits at the desk; same motion as violating X-20. | — | ok |
| X-23 | Cashier locks the laptop, goes to the safe, leaves the room. | T7@12s(0.50) | FALSE ALARM |
| X-25 | Cashier goes to the safe and leaves; desk has no cash. | T3@0s(0.65) | FALSE ALARM |
| X-27 | Cashier goes to the safe and leaves; desk empty (pair of X-26). | — | ok |
| X-29 | Cashier locks the laptop (Windows lock screen), goes to the safe, leaves (pair of X-28). | — | ok |
| X-31 | not inspected | T7@7s(0.50) | FALSE ALARM |
| X-35 | not inspected | — | ok |
| X-37 | not inspected | — | ok |
| X-39 | not inspected | — | ok |
| X-43 | not inspected | — | ok |
| X-45 | not inspected | T9@20s(0.60) | FALSE ALARM |
| X-48 | not inspected | — | ok |
