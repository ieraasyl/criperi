# Detection plan: rules → CV and VLM methods

Scope: how each rule in `rules_kb.json` (27 rules, quotes audited against adilet) can be detected from the cashier-workplace video, using classical CV models and a local VLM. Written against what the pipeline already does (stages 0–5, 8–9) and what the 42 training clips show.

Status legend: **built** = in the pipeline now; **planned** = designed here, not built; **no** = not detectable from video.

---

## 1. Principles

1. **CV finds states and timing; the VLM answers closed questions.** CV (detection, tracking, pose, small classifiers) gives *who is where and what state things are in* at second resolution. A VLM is poor at "when exactly" but good at "what is this object / what is happening between these two people".
2. **The rule engine decides; the rule text is looked up, never generated.** A violation type maps to a rule id in `rules_kb.json`. The VLM never picks or words a norm.
3. **State, not motion.** "Cashier left the desk" occurs in no-violation clips too (X-23, X-25, X-27, X-29). Every type combines the movement with a state: cash on the desk, screen unlocked, client present.
4. **The VLM may abstain.** The bank wants few false alarms (FAQ 16). A candidate the VLM cannot confirm is dropped or down-weighted, not reported.
5. **Output stays at the level of what is visible** (brief §5: "без неподтверждённых предположений"). Weak norms get soft wording.

---

## 2. Architecture

```
video
 └─ decode (4 fps) → layout per camera segment → people (track, pose, role, posture, zone)
      → objects (desk candidates + trained heads) → per-second timeline
           → event rules → CANDIDATES (type, time range, timecode, detector score, facts)
                 ├─ VLM verifier   : 1 closed question per candidate, 4-6 frames + crop
                 ├─ VLM sweep      : 1 frame / 3 s, one yes/no per rule type (finds unseen types)
                 └─ fusion         : score = f(detector, duration, VLM) vs per-type threshold
                       → verdict, evidence frames, templated description + rule citation
```

### Where each model sits

| Role | Model | Training |
|---|---|---|
| People | RT-DETRv2 r18 + ByteTrack | none |
| Posture and hands | box-height ratio vs seated height; RTMPose wrists | none |
| Scene layout and desk candidates | Grounding DINO tiny, text prompts | none |
| State classification (object category, screen, safe) | SigLIP embeddings + logistic-regression heads | **yes: hand-labeled crops** |
| Verification and description | Qwen2.5-VL-7B or Qwen3-VL-8B (local, GPU) | none (prompting) |

---

## 3. Per-rule detection methods

Priority: 1 = build first, 4 = last or skip. "Clips" = training clips where it appears (confidence A/B from my analysis).

### 3.1 Rules we can detect (build)

#### R-CASH-LOCKED: НБРК № 231 п. 20 ч. 2 (types T1, T7). Priority 1

> «Все ранее принятые наличные деньги хранятся в специальных устройствах … закрывающихся на ключ»

Clips: X-26 (A), X-15 (A), X-4, X-13, X-38 (B).

| | |
|---|---|
| **Violation** | Cash or a cash bag lies on the desk while the cashier is away, or while a new visitor arrives with the previous client's cash still out. |
| **CV signals** | Object candidates on the desk (Grounding DINO on the desk crop) → category head `cash / cash_bag` → `on_desk` and not `in_hand` (wrist distance) → dwell time. Cashier **stand-up** from the seat (box height > 1.35× seated) dates the event. Visitor arrival (new track reaching the desk zone). |
| **Event rule** | `cash_on_desk` ∧ cashier away ≥ 2 s → timecode = stand-up second (cash may be detected up to 4 s later). Or new visitor arrives with cash on desk and no client in the previous 3 s. |
| **VLM question** | "Frames show a cashier's desk. Is there money (banknotes) or a cash bag lying on the desk while the cashier is not sitting at it? Answer `{confirmed, confidence, observation}`." |
| **VLM adds** | Rejects false cash (paper, ID cards, phone); resolves bag vs. other objects. |
| **Hard parts** | Cash is tiny (≈25×12 px at 464×848). Needs the trained head and crops at native resolution for the VLM. |
| **Data needed** | ~300 crops labeled cash / cash bag / not cash. |
| **Status** | CV built (zero-shot SigLIP, noisy); heads and VLM planned. |

#### R-PERSONAL-ITEMS: НБРК № 231 п. 12 (types T2, T3, T9). Priority 1 (T2), 3 (T3, T9)

> «Хранение личных вещей, наличных денег и ценностей работников банка осуществляется вне кассового узла»

Clips: X-40 (A, coat and handbag), X-46 (A visually, cup and food; weak norm).

| | |
|---|---|
| **Violation** | Coat, bag, food, drink or phone at the cash workstation. |
| **CV signals** | Same desk-object pipeline, categories `personal`, `food_drink`, `phone`; dwell ≥ 3 s; hand-to-mouth gesture from wrists for drinking. |
| **VLM question** | "Is there a personal item (coat, handbag, backpack) on the work desk?" / "Is the cashier eating, drinking, or holding a phone?" |
| **Wording** | T2 direct. T3 and T9 soft: the act names belongings, not food; a phone's camera function cannot be seen. |
| **Data needed** | Crops labeled personal / food-drink / phone / work item. |
| **Status** | CV built (zero-shot, noisy); heads and VLM planned. |

#### R-IS-LOCK: АРРФР № 53 п. 71 пп. 9 (type T5). Priority 2, **weak norm**

> «при неактивности пользователя более 30 (тридцати) минут информационная система автоматически … блокирует рабочую станцию или ноутбук…»

Clips: X-28 (A), X-11 (A).

| | |
|---|---|
| **Violation** | Cashier leaves the desk while the laptop still shows the work application (the controls X-23 and X-29 lock the screen first). |
| **CV signals** | Laptop box from layout → crop → `screen` head: `work_screen / locked / off`. Cashier away ≥ 3 s with `work_screen` throughout. |
| **VLM question** | "Is the laptop screen showing an application window, or a lock screen / wallpaper, or is it off?" |
| **Wording** | "Screen left unlocked when leaving"; never "violates п. 71" (the clause sets a system auto-lock, not an employee duty). |
| **Limits** | In 20 of 22 violation clips the laptop is dark, so this fires in only ~2 clips. Needs organizer confirmation of the intended norm. |
| **Status** | CV built; head and VLM planned. |

#### R-LEAVE: НБРК № 231 п. 22 (type T6). Priority 2

> «Кассовому работнику не допускается покидать рабочее место до окончания обслуживания клиента.»

No clip shows it (in every leaving clip no client is present), but the hidden test set may.

| | |
|---|---|
| **CV signals** | Cashier leaves the room (zone doorway/elsewhere/absent) ≥ 3 s while a visitor is seated at the desk the whole time. |
| **VLM question** | "A person sits at the desk and the cashier has walked away. Is the client still waiting at the desk?" |
| **Status** | CV built; untested on real clips. |

#### R-ENTRY-ISOLATION / R-VAULT-ACCESS: НБРК № 14 п. 8; № 231 п. 75 (type T4). Priority 2, **indirect**

Clip: X-30 (B).

| | |
|---|---|
| **Violation** | A non-cashier stands on the cashier's side of the desk or at the safe. |
| **CV signals** | Visitor foot-point within 0.6 body heights of the cashier's seat (feet, not box centres, to avoid perspective error). Excludes the client seated across (calibrate on X-1/X-4 vs X-30). |
| **VLM question** | "Is a person other than the cashier standing behind the desk next to the cashier?" |
| **Wording** | Neutral: "outsider on the cashier's side"; п. 8 regulates premises design, not who may enter. |
| **Status** | CV built; threshold needs calibration. |

#### R-CASH-LOCKED, safe variant (type T7). Priority 3

Safe door `open` (crop head) while the cashier is out of the room ≥ 3 s. VLM: "Is the safe door open?" Weak evidence in the data (the safe flaps open/closed in zero-shot).

### 3.2 Rules we can only partly detect (plan, lower priority)

| Rule id | Norm | Violation | CV method | VLM check | Priority |
|---|---|---|---|---|---|
| R-COUNT-CLIENT | № 231 п. 20 ч. 1 | Counting while the client is absent | Visitor track present at the desk during hand-count; X-4 shows the client stepping out | "Is the client at the desk while the cashier counts?" | 3 |
| R-COUNT-ISSUE-CLIENT | № 231 п. 31 пп. 4 | Issued cash counted without the client | Same logic during issue | same | 3 |
| R-CLIENT-COUNTS | № 231 п. 35 | Client leaves before counting received cash | Visitor leaves the desk zone right after receiving cash | "Did the client count the money at the desk?" | 4 |
| R-CONTROLLER | № 231 п. 42 | Counting alone | Only one person in the zone while the counting machine runs; X-44 | VLM cannot judge a second person's role | 4 |
| R-HANDOVER-DOC | № 231 п. 15 | Cash passed between cashiers, no document | Wrist proximity between two staff tracks | "Is a bundle handed between two staff?" (paper not visible) | 4 |
| R-DOUBTFUL-ACT | № 231 п. 59 | Doubtful note handled alone | Hand-off of one note, no second employee | hard at this resolution | 4 |
| R-COLLECTOR-CREW / KIT / CHECKS-BAG / ESCORT-CREW | № 231 пп. 98, 103, 107, 127 | Collector alone or without armour, bag not checked | Person count carrying a bag; vest/helmet attribute | "Is the person wearing body armour / a helmet?" | 4, unverified; X-17, X-18 to be checked |
| R-CASH-ROOM-DOOR | № 14 п. 16 пп. 1 | Door open or propped | Door box state (open/closed head) | "Is the door open?" | 4 |
| R-IS-NO-SHARED-ACCOUNT | № 53 п. 73 пп. 5–6 | Another person works at the cashier's laptop | Non-cashier wrist near the laptop box | "Is someone other than the cashier typing?" | 4, not seen in any clip |
| R-KEYS, R-VAULT-ACCESS (keys) | № 231 пп. 76, 80 | Key left in the lock or on the desk | Small-object detection (key) | hard | 4 |
| R-CLIENT-ID | № 231 п. 19, 31 пп. 1 | No ID check | Absence of a "document handled" moment | unreliable; do not report | skip |

### 3.3 Rules not detectable from video (do not claim)

R-EOD-RECON (№ 231 п. 56), R-VAULT-STORAGE (п. 74), R-IS-PERSONAL-ACCOUNTS (№ 53 п. 40), R-CAMERAS (№ 14 п. 16 пп. 6; a precondition for the footage, not a cashier act), R-COUNT-EQUIPMENT (rejected: control X-3 counts by hand too), and the items in `not_checkable_from_video`: balance reconciliation, shortage acts, audit, signatures and receipts, due-diligence checks, vehicle rules, premises construction. The brief forbids claiming violations that need invisible data.

---

## 4. VLM design

### 4.1 Model and serving
- Qwen2.5-VL-7B-Instruct or Qwen3-VL-8B, local weights, GPU ≥ 24 GB, fp16/bf16. Verify the license **for the exact size** (the 3B size is non-commercial; FAQ 24 requires commercial use).
- Serving: `transformers` with batched generation, or vLLM if available on the host. No network at inference.
- Cost control: the VLM only sees candidates (typically 1–5 per video) plus the sweep (≈50 frames at most).

### 4.2 Three jobs

1. **Verify a candidate.** Input: 4–6 frames from `[timecode − 2 s, timecode + 4 s]` at native resolution (464×848 is small; do not downscale), one enlarged crop of the relevant region, and the CV facts as text. One closed question per type. Output JSON `{confirmed: true|false|unknown, confidence: 0..1, observation: "<one sentence of what is visible>"}`.
2. **Sweep for unseen types** (FAQ 9). One frame every ~3 s, a fixed checklist with one yes/no question per type in the rule base. Any "yes" without a CV candidate becomes a low-confidence candidate and goes through job 1. The sweep **can only raise types that have a rule to cite**; "something looks wrong" is never reported.
3. **Write one observation sentence** for the description. The rest of the description is a template filled from timeline facts, and the rule quote comes from the KB.

### 4.3 Prompt skeleton (per type)

```
You are checking a surveillance video of a bank cashier's workplace from above.
Facts measured automatically: {facts}.
Look only at what is visible in the frames. Do not guess.
Question: {closed question for the type}
Answer strictly as JSON: {"confirmed": true|false|"unknown", "confidence": 0-1,
"observation": "<one sentence, what is visible>"}.
Use "unknown" if the frames do not show it clearly.
```

Closed questions (English; translate to Russian if it improves accuracy):

| Type | Question |
|---|---|
| T1 | "Are banknotes or a cash bag lying on the desk while the cashier is not at the desk (or a different person has just arrived)?" |
| T2 | "Is there a personal item (coat, handbag, backpack) on the cashier's work desk?" |
| T3 | "Is the cashier eating, drinking, or is there food or a drink on the desk?" |
| T4 | "Is a person other than the cashier standing on the cashier's side of the desk?" |
| T5 | "Does the laptop screen show an application window (not a lock screen, wallpaper, or a black screen)?" |
| T6 | "Is a person waiting at the desk while the cashier is out of the room?" |
| T7 | "Is the safe door open?" |
| T9 | "Is there a mobile phone on the desk or in the cashier's hand?" |

### 4.4 Fusion

```
score = w_det · detector_score + w_dur · duration_factor + w_vlm · vlm_confidence(confirmed)
if vlm == "unknown":   score *= 0.7      # not contradicted, not confirmed
if vlm == false:       score *= 0.2      # contradiction: drops below threshold
report if score ≥ threshold[type]
```

Weights and thresholds are fitted on the training clips. Calibrate in this order: first **zero misses** on the training violations, then minimize false alarms on the controls (FAQ 16).

### 4.5 Failure handling
- Hard cutoff at 4 minutes returns the CV-only result.
- A VLM timeout or invalid JSON counts as "unknown", never as confirmation.
- Same input gives the same output (temperature 0), so runs are reproducible.

---

## 5. Data and training needs

| Item | Size | Purpose |
|---|---|---|
| Crops: object category | 300–600 | cash / cash bag / personal / food-drink / phone / other (`kassa export-crops`) |
| Crops: laptop screen | ~150 | work screen / locked / off |
| Crops: safe | ~150 | open / closed |
| Clip-level event labels | 42 clips | type, start/end second, matching control (resolve the 12 unexplained clips) |
| Own footage | 2–3 new angles, staged violation + correct version | the only real test of angle robustness (10 points) |

Train heads with clips as groups in cross-validation (neighbouring seconds of one clip look alike). Augment with perspective warp, scale, blur, brightness and JPEG compression to imitate another camera.

---

## 6. Evaluation

- `kassa eval` runs all clips with per-stage caching and reports verdict confusion, timecode error, per-type counts, threshold sweep, and runtime.
- Add VLM on/off comparison: false alarms and misses with CV only vs. CV + VLM.
- Hold out own-angle footage and flipped / cropped / rescaled variants as the robustness set; never tune on it.
- Per-type acceptance targets (proposal): T1 and T2 recall ≥ 90 % on A/B clips with ≤ 1 false alarm on the 20 controls; timecode within ±3 s on ≥ 70 % of detected violations.

## 7. Risks

| Risk | Mitigation |
|---|---|
| 12 of 22 violation clips unexplained | Resolve as a team first; report the gap |
| Cash is tiny; zero-shot is noisy | Train the head; give the VLM native-resolution crops |
| Different test angle breaks zones and distances | Foot-point and posture-ratio logic, per-video layout, own-angle tests |
| VLM hallucinations | Closed questions, "unknown" allowed, contradiction drops the score |
| Weak norms (laptop, outsider, food) cost grounding points | Soft wording, lower confidence, organizer questions |
| Runtime: pipeline ≈ 2× video length on an M4 before the VLM | GPU host; detect objects only near events; fewer layout frames |

## 8. Build order

1. Evaluation harness (done) → per-stage cache, report.
2. Resolve clip labels (step 2 of the plan) in `labels/clips.yaml`.
3. Heads for cash / personal items (T1, T2), then screen (T5), then safe (T7).
4. VLM verifier for T1 and T2 only; compare with CV-only in the harness.
5. Extend verifier to T5, T4, T6; add the sweep.
6. Fusion and threshold calibration; own-angle robustness runs.
7. Lower-priority partial rules only if time remains.
