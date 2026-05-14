# Design A — Late-fusion concat (closed, ✗ FAIL)

## Aim

Falsify whether the cheapest possible merge — concatenate canonical c2hgi
check-in embeddings with HGI POI embeddings at the *input layer* of the
downstream model — is sufficient to lift reg without harming cat.

## Mechanism

Per-step input becomes `[c2hgi_checkin_emb(64) ‖ hgi_poi_emb(64)] = 128-dim`.
No engine retrain; both substrates already exist. Downstream `next_gru`
(cat) and `next_getnext_hard` (reg) consume the wider input.

## AL/AZ leak-free results

| State | cat F1 | Δ vs canonical | reg Acc@10 | Δ vs canonical |
|---|---:|---:|---:|---:|
| AL | 32.27 ± 1.25 | −8.49 pp | 40.43 ± 2.81 | −18.72 pp |
| AZ | 31.69 ± 0.38 | −11.52 pp | 41.08 ± 2.50 | −9.16 pp |

Both states 5/5 folds in the wrong direction (Wilcoxon p=0.0312 against the
hypothesis of non-inferiority).

## Verdict

✗ FAIL on both cat and reg at both states. Late-fusion at the input is
strictly worse than either single substrate. Likely cause: the head cannot
suppress the wide-low-signal HGI columns when computing the cat task, and
the reg head sees a noisier per-step input than HGI alone provides.

Closed 2026-05-04. Designs B/E/F/G/H were spawned to recover what late
fusion broke.
