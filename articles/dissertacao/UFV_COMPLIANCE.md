# UFV_COMPLIANCE.md — norms, formatting, prerequisites, and the submission pipeline

> Everything the dissertation must comply with, distilled from the primary sources and
> **re-verified against the live pages on 2026-07-18**. Supersedes
> `articles/[TESE]_MTL_POI/NORMAS_UFV.md` (June 2026 extraction) — one material change since
> then: a **new PPGCC regimento interno approved 2026-07-09** (§3 below). The local copy of the
> submission manual is [`docs/Manual-de-entrega-de-dissertacoes-e-teses-04_2026.pdf`](docs/).

---

## 1 · The two-build reality (understand this first)

There are TWO deliverable shapes, and they differ:

1. **Defense build** (to secretariat + banca, ≥20 days before the defense): a conventional full
   PDF — cover page, approval sheet, lists, sumário, body — like the Viegas example PDF. This is
   what the banca reads.
2. **Final AcademicoPG build** (after approval, within 3 months): the system generates/holds the
   cover, ficha catalográfica, dedicatória, agradecimentos, epígrafe, resumo, abstract and
   keywords (filled in web forms, "colar sem formatação"); the **uploaded PDF contains ONLY**, in
   order: lista de ilustrações / tabelas / siglas / símbolos (all optional) → apresentação
   (optional) → **sumário (mandatory)** → **body (mandatory)**. Page numbering starts on the
   first body page; the SYSTEM's emitted PDF (RASCUNHO watermark) is the authoritative numbering
   reference.

The LaTeX setup must produce both from one source ([`TEMPLATE.md`](TEMPLATE.md) §3).

## 2 · Mandatory formatting (Manual 04_2026 §7–§8)

| Item | Rule |
|---|---|
| Font | **Arial or Times New Roman, size 12** (note: Viegas used Palatino and passed — but that predates strict enforcement; comply, don't gamble) |
| Paper | A4 (21 × 29.7 cm) |
| Margins | top + left **3 cm**; bottom + right **2 cm** |
| Line spacing | **1.5** |
| Page numbers | top-right corner, arabic; **start on the first body page (introduction)** |
| Pre-textual pages | **counted but not numbered**; cover + ficha catalográfica neither counted nor numbered |
| Numbering example | 10 pre-textual pages → the first body page is numbered 11 |
| Resumo/Abstract (system fields) | text block only, no paragraphs, no header (system generates the header); keywords one per line, lowercase except proper nouns, no punctuation |
| Agradecimentos (system field) | the CAPES (Financiamento 001) + FAPEMIG + CNPq funding sentence is auto-inserted and cannot be altered (can be reordered among other thanks) |

Older Normas-gerais print-era rules (≥40 mm binding margin etc.) are superseded by the manual
for the final build; SI units and legibility rules (§3.x) still apply.

## 3 · Coletânea + defense prerequisites (Normas gerais rev. 11/10/2019 + regimento 2026-07-09)

**Coletânea format (Normas §2.3, §2.6 — confirmed verbatim on the live PDF):**
- Body as "artigos científicos … **publicados, aceitos, ou submetidos** para publicação" →
  MobiWac as a submitted paper is eligible.
- Structure: **(i) Introdução Geral, (ii) Artigo(s), (iii) Conclusão Geral**; Introdução/Conclusão
  may carry their own bibliographies; articles have **free formatting** given internal
  consistency (§2.6); different formattings (§2.6.2) and **different languages (§2.6.3)**
  admitted; previously published articles may be reproduced from the originals (§2.6.4).
- Language of the work: PT, EN, or ES **at the Comissão Orientadora's discretion** (§1.3) —
  the English frame needs the advisor's OK, nothing more.
- Optional closing sections ("Perspectivas Futuras" etc.) allowed (§2.7); post-textual
  apêndices/anexos allowed (§2.8).

**Defense prerequisites (PPGCC regimento interno — ⚠ NEW VERSION approved 2026-07-09):**
- **Art. 21 §1 (new wording):** master's student is apt to defend only after proof of
  **publication or acceptance of ≥1 article from the research in a scientific event**, OR
  **proof of submission to a CS journal**, with minimum quality per an internal resolution
  (CAPES-Computação-aligned). The word "Qualis" no longer appears — BUT the still-linked
  June/2024 defense checklist demands "Qualis A4 or superior" proof. **STATUS 2026-07-18: the
  substance is covered — CBIC is published with DOI `10.21528/CBIC2025-1191324` (verified;
  CoUrb `10.5753/courb.2026.22960` is a second published article as backup). Remaining action:
  file the comprovante with the secretariat (ppgcc@ufv.br) and confirm the operative checklist
  bar.**
- **Art. 22:** dissertation text to the secretariat **≥20 days before the defense date**.
- **Art. 23:** public defense, presentation up to 50 min, then arguição.
- **Anti-plagiarism certificate** (UFV institutional tools) is mandatory — "a defesa não será
  aprovada" without it (defense checklist item; also relevant to the AI-use disclosure,
  [`AGENT_GUARDRAILS.md`](AGENT_GUARDRAILS.md) §6).
- Wet-signature items survive the online flow: termo de assentimento (checklist item 12) and
  the BBT authorization term — plan physical signatures around the defense.

## 4 · Post-defense pipeline (Manual 04_2026, step by step)

1. Corrections per the banca → final text.
2. AcademicoPG → "Entrega de tese/dissertação" → fill pre-textual forms (**colar sem
   formatação**; system does not auto-create paragraphs).
3. Upload the body-only PDF (§1 build 2).
4. "Emitir PDF para solicitação da ficha catalográfica" → RASCUNHO-watermarked draft → **check
   numbering against the system PDF** (it is authoritative; adjust the LaTeX page-counter offset).
5. Send the RASCUNHO to BBT (https://www.bbt.ufv.br/ficha-catalografica-on-line/) → ficha ready
   in ~7 days (email notification) → attach it in the system.
6. "Visualizar documento completo" → final check → "Enviar" (password = digital signature) →
   advisor approves → PPG homologation → diploma request automatic.
7. **Deadline: 3 months after the defense**, fine afterwards (monthly, PagTesouro); the 7-day
   ficha SLA is inside the window. If the advisor rejects, the clock restarts from the defense
   date (fine risk) — get the text right before "Enviar".

## 5 · Timeline flags for an August 2026 defense (today: 2026-07-18)

- Defense in the window **Aug 18–29** ⇒ **text + banca locked between Jul 29 and Aug 9**
  (Art. 22's 20 days + banca formed in AcademicoPG + members informed).
- Before the defense is approved: anti-plagiarism certificate + Art. 21 proof on file.
- After an August defense: final AcademicoPG deposit due by **~late November 2026**.
- Full schedule and fallback: [`PLAN.md`](PLAN.md).

## 6 · AI-use policy state (2026-07-18; details in AGENT_GUARDRAILS §6)

- No binding UFV/PPGCC norm on AI-assisted writing found. UFV/DPE published recommended
  disclosure guidelines (03/2026). **CNPq Portaria nº 2.664/2026** mandates AI-use declaration
  for CNPq-linked researchers (tool + purpose; AI-generated content as human-authored is
  vedado). CAPES converging via GT/technical notes (primary texts partially unverified).
- Practical ruling: include a disclosure note (AGENT_GUARDRAILS §6 D1) and raise it with the
  advisor early.

## 7 · Verified sources (all HTTP-checked 2026-07-18)

- Manual de entrega 04_2026 (PPG):
  https://www.ppg.ufv.br/wp-content/uploads/2026/04/Manual-de-entrega-de-dissertacoes-e-teses-04_2026.pdf
  (local copy in [`docs/`](docs/))
- Normas gerais de Teses e Dissertações (rev. 11/10/2019):
  https://www.ppg.ufv.br/wp-content/uploads/2012/08/Normas-gerais-de-Teses-e-Dissertac%CC%A7o%CC%83es-12.pdf
- PPGCC Regimento interno (aprovado 09/07/2026): https://ppgcc.ufv.br/regimento-interno/
- PPGCC Procedimentos finais (Word models PT/EN, checklists): https://ppgcc.ufv.br/procedimentos-finais/
  - Word model EN: https://ppgcc.ufv.br/wp-content/uploads/2024/07/UFV-2019-Modelo-dissertacao-tese-em-ingles-7-1-2.docx
  - Pre-textual checklist: https://ppgcc.ufv.br/wp-content/uploads/2024/07/CHECK-LIST-%E2%80%93-Formatacao-das-Paginas-Pre-Textuais-Dissertacao-e-Tese.pdf
  - Defense checklist (Jun/2024 — predates the new regimento, Qualis wording likely stale):
    https://ppgcc.ufv.br/wp-content/uploads/2024/07/CHECK-LIST-PARA-DEFESA-DE-DISSERTACAO.pdf
- PPG signature pages + Resumo/Abstract models:
  http://www.ppg.ufv.br/wp-content/uploads/2012/08/Modelo-pgs-de-assinaturas.pdf ·
  http://www.ppg.ufv.br/wp-content/uploads/2012/08/Modelo-Resumo-e-Abstract1.pdf
- BBT: ficha catalográfica https://www.bbt.ufv.br/ficha-catalografica-on-line/ · Normalização
  2025 https://www.bbt.ufv.br/wp-content/uploads/2025/02/Normalizacao-de-trabalhos-academicos-2025-UFV.pdf
- AI policy: UFV/DPE guide
  https://tecido.dpe.ufv.br/wp-content/uploads/2026/03/IA-em-Pesquisa-Educacional_-Recomendacoes-1.pdf ·
  CNPq Portaria 2.664/2026 announcement
  https://www.gov.br/cnpq/pt-br/assuntos/noticias/cnpq-em-acao/cnpq-publica-portaria-que-institui-politica-de-integridade-na-atividade-cientifica

**Open items (carried into PLAN.md):** (a) operative Art. 21 quality bar + which regimento
governs pre-2026 enrollees — ask the secretariat (substance already covered, §3); **Qualis
strata now known** (deep-research, 2026-07-20, 3-vote adversarial verify, primary source =
CAPES "Relatório Qualis Eventos 2017-2020, Computação"): **CBIC = B4, CoUrb = B4, MobiWac = B2**
— all three classified directly by CAPES's own Computação area coordination (not by an
Engineering/Automação committee, despite CBIC being organized by ABRICOM/SBIC/SBA). This is the
most recent complete nominal listing found; whether it still governs 2025/2026 (vs. a newer
"Qualis Eventos 2025" Sucupira cycle, unconfirmed per-venue) and whether B4/B2 actually clear
the Art. 21 "resolução interna" bar (not itself located) is **still the open question for the
secretariat** — full findings + sources in
[`docs/research/qualis_classification_2026-07-20.md`](docs/research/qualis_classification_2026-07-20.md);
(b) ~~CBIC proceedings entry/DOI~~ RESOLVED 2026-07-18 (`10.21528/CBIC2025-1191324` verified);
(c) CAPES NT 3/2025 primary text.
