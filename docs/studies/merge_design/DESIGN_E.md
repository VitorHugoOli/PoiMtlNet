# Design E — Per-task projector heads with reg stop-grad (closed, ✗ cat fail)

## Aim

Test the advisor's hypothesis that the cat-vs-reg trade-off in
POI2Vec-augmented c2hgi is **gradient-interference**, not a representational
limit. If true, splitting the heads into per-task projectors with
stop-gradient on the reg branch should restore cat without sacrificing reg.

## Mechanism

POI2Vec is concatenated to check-in features at the input (the failing
"probe"). Two small projector heads (Linear → ReLU → Linear) are added —
one for cat, one for reg. **Stop-gradient is applied between the reg
projector and the shared encoder**, so the reg path's gradient cannot
propagate back through the encoder. Cat path gradients flow normally.

## AL/AZ leak-free results

| State | cat F1 | Δ vs canonical | reg Acc@10 | Δ vs canonical |
|---|---:|---:|---:|---:|
| AL | 31.22 ± 0.93 | **−9.54 pp** (TOST p=0.9999) | 61.24 ± 3.99 | +2.09 pp |
| AZ | 33.83 ± 0.38 | **−9.38 pp** (TOST p=1.0000) | 52.62 ± 2.99 | +2.38 pp |

fclass linear probe: 90.97 / 93.16 (close to HGI 98%).

## Verdict

✗ FAIL on cat dominance — −9 pp at both states, identical magnitude to the
unprojected probe. Reg gain matches HGI within σ.

The interpretation: the cat collapse is **not** gradient-interference from
the reg head — it's already locked in by the time the input concat reaches
the encoder. POI2Vec injection at the *input* is intrinsically incompatible
with cat. This finding redirected the ladder away from input-level fusion
and toward POI-boundary fusion (Designs B/H/I/J/M).
