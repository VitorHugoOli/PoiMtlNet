# MobiWac 2026: Glossary and Plain-Word Guide (working draft v1)

> **Why this file exists.** Two of the three BRACIS reviewers rejected the previous paper mainly because it was
> hard to follow: too many acronyms and too much machine-learning jargon, used without explanation. MobiWac
> readers know networking, not embeddings, so the paper must use simple words by default and spend its few
> technical terms carefully.
>
> **v1 change.** Trimmed to only the terms the [`PAPER_PLAN.md`](PAPER_PLAN.md) actually uses. Task names are
> **category** and **region** (the literature terms in CBIC/CoUrb and the field), not activity/area. Unused
> networking acronyms (AP, MEC, 5G) and unused ML jargon were removed. Vitor will revise; it leans toward
> simpler, push back where a term earns its place.

---

## 1 · Canonical names (commit to these; use them consistently)

| Concept | Use this | Technical term (first use only) | Never use |
|---|---|---|---|
| The "what" task | **next category** / **category prediction** | next-category | "activity", "POI classification", "next-POI category" |
| The "where" task | **next region** / **region prediction** | next-region | "area", "location", "next-POI" |
| A place | **place** / **point of interest (POI)** | POI | "venue" |
| One region unit | **region** (a **census tract**; a **mahalle** in Istanbul) | — | "cell", "zone" |
| One visit | **check-in** | — | "event" |
| Our representation | **check-in-level representation** (Check2HGI) | Check2HGI | "substrate" (repo word) |
| Place-level baseline | **place embedding (HGI)** | HGI | "the baseline" alone |
| One model, both tasks | **the joint model** / **single multi-task model** | MTL model | naming "MTLnet" before introducing it |
| One task, one model | **dedicated model** / **single-task model** | STL | "baseline" alone |

> **Naming rule (from the literature audit):** use **category** and **region**, never **activity** or **area**.
> "Activity" appears only when describing other papers (MCARNN, iMTL) where the field's older term was
> "next activity". "Region" is the canonical spatial term; name the unit (census tract; mahalle).

> **Keep three targets distinct:** *next category*, *next region*, and *next place* (the exact POI). The last
> submission blurred them. We predict the first two and not the exact next place; say so once, early.

> **Honesty rule (region wording, updated 2026-06-24):** the region result is now **mixed**: a **beat**
> (superiority, paired Wilcoxon) at the large region counts, and a **match** at the small. The plain words
> "matches" or "stays within two points" are fine in the narrative, **but every formal claim sentence (abstract,
> §6.2, §7, the contribution bullets) must carry "statistically non-inferior within a two-point margin (TOST)" at
> least once** for the small-state match. Never "ties" or "Pareto-dominates", and never "beats region everywhere"
> (the small states are matches, slightly negative).

> **Hyphenation:** hyphenate the compound adjective before a noun ("next-category prediction"); leave the bare
> task name open ("the next category"). Same for next-region and next-place.

---

## 2 · Acronyms (expand on first use, then short form; keep the count low)

| Acronym | Expansion | Plain meaning | First-use plan |
|---|---|---|---|
| **LBSN** | Location-Based Social Network | An app where people share the places they visit. | Expand in the abstract/intro. |
| **POI** | Point of Interest | A place a person can visit. | Expand in §1. |
| **MTL** | Multi-Task Learning | Training one model to do several jobs at once. | Expand in §1; then "the joint model" mostly. |
| **STL** | Single-Task Learning | One model per job (the usual way). | Expand once as "dedicated / single-task model". |
| **macro-F1** | balanced F1 score | A score that counts every category equally, so rare ones matter. | Gloss once. |
| **Acc@10** | accuracy at top 10 | How often the true region is among the model's top ten guesses. | Define once. |
| **CV** | cross-validation | The way we split the data to test fairly. | "5-fold cross-validation" once, then CV. |

**Introduce once, then mostly avoid (method internals):**

| Acronym | Expansion | Use note |
|---|---|---|
| **HGI** | Hierarchical Graph Infomax | The standard place embedding we build on. Expand plus one-line gloss (§2.1). |
| **DGI** | Deep Graph Infomax | HGI's ancestor. Mention once in the §2.1 primer. |
| **Check2HGI** | (our method's name) | Introduce once as "our check-in-level representation (Check2HGI)", then prefer the plain phrase. |
| **CTLE / STAN** | (baseline names) | Name once each, in related work and the baseline list. |

**Do not use in the paper (repo-internal):** `C2HGI` (write "Check2HGI"), recipe and version codenames (B9,
v11–v16, champion-G, H3-alt; write "our model"), `log_T` (write "region-transition prior"), internal finding IDs
(C25, CH16, CH19, F-numbers).

---

## 3 · Jargon to plain substitution (keep / gloss / avoid)

| Jargon | Say instead | Verdict |
|---|---|---|
| embedding / representation | **a vector that summarizes a place or visit** | gloss once, then "representation" |
| per-visit / contextual | **each visit gets its own vector** (not one fixed vector per place) | keep; this is our key idea |
| substrate | **the representation** | avoid (repo word) |
| graph | **a network of places linked by similarity, time, and distance** | keep, gloss once |
| infomax objective | **trains the vectors to tell real neighborhoods from shuffled ones** | gloss; skip the formula unless space allows |
| hierarchical | **organized as check-in → place → region → city** (four levels; ours adds the check-in level) | keep |
| multi-task / parameter sharing | **one model doing two jobs and sharing most of its parts** | keep, gloss "sharing" once |
| single-task ceiling | **the best a dedicated, one-job model reaches** | keep; "dedicated model" |
| negative transfer | **sharing hurts one task** | gloss; use the plain phrase |
| non-inferiority (TOST) | **statistically no worse than, within a stated margin** | gloss once with the 2-point margin |
| Markov / transition baseline | **a simple "what usually follows what" baseline** | keep, gloss once |
| overlapping (stride-1) windows | **overlapping windows of recent visits** | gloss once (nine visits plus the next as target) |
| transductive | **the representation was trained seeing all places** | gloss only where the leak discussion needs it |
| ablation | **a controlled test that removes one piece** | keep, gloss once |
| seed / fold | **repeated runs / data splits** | "averaged over runs and splits" |

---

## 4 · Words to avoid or always explain

- **"activity" / "area"** for the two tasks: use **category** / **region**.
- **"Pareto", "Pareto-dominate"**: avoid; and recall we cannot even claim it. Say "better on one task without
  being worse on the other" only if true.
- **architecture names** ("cross-attention", "FiLM", "residual block", "transformer"): describe what they do in
  plain words; name one only if it is truly load-bearing.
- **our internal research words** ("substrate", "engine", "head", "regime", "frontier", "orthogonal gradients"):
  jargon in the paper. Say "representation", "model", "output", "setting".
- **recipe / version codenames** (B9, v11–v16, champion-G): invisible to the reader; say "our model".
- **"SOTA"**: write "state of the art" (never "SOAT").
- **undefined metrics**: never give a number without its reference point (majority-class or Markov floor).
- **dense tables with no lead sentence**: every table gets a "read this as" sentence.

---

## 5 · Numbers in plain language (say it once, this way)

| We report | Say it as | Reference point |
|---|---|---|
| macro-F1 (category) | "out of 100, higher is better; counts each of the 7 categories equally" | majority-class floor; HGI baseline |
| Acc@10 (region) | "how often the true next region is in the model's top 10" | Markov-1 floor |
| Δ (delta) | "the gain (or cost) versus the dedicated model" | always paired, with the margin |
| non-inferior (2-point margin) | "no worse than the dedicated model by more than 2 points, with statistical support" | state the margin every time |

---

## 6 · Consistency checklist (before submission)

- [ ] Every acronym expanded on first use; acronym count as low as possible.
- [ ] **category / region** used throughout; never "activity" / "area".
- [ ] **next category / next region / next place** kept distinct; "we do not predict the exact next place" stated once.
- [ ] Region wording: "beats" at the large counts (superiority) and "matches / non-inferior within two points"
  at the small; every formal claim sentence carries "statistically non-inferior within a two-point margin (TOST)"
  at least once; never "ties", "Pareto", or "beats region everywhere".
- [ ] No recipe or version codenames anywhere.
- [ ] No bare "substrate / engine / head / cross-attention"; replaced or glossed.
- [ ] Every table has a "read this as" lead sentence.
- [ ] "state of the art", never "SOTA".
- [ ] American English throughout (behavior, modeling, neighbor, favor).
- [ ] No em-dash ("—"); use commas, parentheses, semicolons, or short sentences.
