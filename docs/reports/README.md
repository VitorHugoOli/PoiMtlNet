# Reports — Orientation for Future Agents

This folder contains meeting reports summarising the project's current state,
key findings, and open discussion points. Reports are written in **pt-br** and
target a mixed research audience (not exclusively ML engineers).

## Naming convention

```
report_v{N}_{YYYYMMDD}.md
```

- `v{N}` — version number, increment when a new report supersedes the previous.
- `{YYYYMMDD}` — creation date.

Example: `report_v1_20260415.md`

## What belongs here vs elsewhere

| Content | Where |
|---------|-------|
| Meeting-ready summary of findings | `docs/reports/` ← here |
| Raw experiment numbers and analysis | `docs/studies/fusion/results/` |
| Active study plan and phase details | `docs/studies/fusion/` |
| Architecture and project overview | `docs/studies/fusion/KNOWLEDGE_SNAPSHOT.md` |
| Archived (pre-bugfix) material | `docs/archive/` |

## Writing guidelines for new reports

- **Language:** pt-br throughout.
- **Tone:** accessible — assume the reader is a researcher, not an ML engineer.
  Avoid implementation jargon (optimizer names, layer counts, etc.) unless
  briefly explained in-line.
- **Structure:** titled topics, each with a short paragraph. Bullet lists for
  outcomes. Cross-reference the analysis doc (link to `docs/studies/fusion/results/`).
- **Length:** aim for something that can be scanned in 5 minutes and read
  thoroughly in 15.
- **Pending items:** mark open questions with `> **Em aberto:**` blockquotes so
  they stand out for discussion.
