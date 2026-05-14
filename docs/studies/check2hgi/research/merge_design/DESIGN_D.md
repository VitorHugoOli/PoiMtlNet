# Design D — Heterograph (⚠ cat lift is leak artifact)

## Aim

Replace the cascade `check-in → POI → region` of c2hgi with a single
heterogeneous graph where check-ins **and** POIs are first-class node
types. Edges: check-in→check-in (sequence), check-in→POI (visits,
bidirectional), POI→POI (Delaunay). Typed encoder weights per edge type.

## Mechanism

See `../DESIGN_D_HETEROGRAPH.md` for the full architecture writeup.

## AL/AZ leak-free results

| State | cat F1 | Δ vs canonical | reg Acc@10 | Δ vs canonical |
|---|---:|---:|---:|---:|
| AL | 72.88 ± 0.80 | **+32.12 pp** ⚠ | 62.23 ± 3.77 | +3.08 pp |
| AZ | 74.73 ± 1.18 | **+31.52 pp** ⚠ | 52.95 ± 2.95 | +2.71 pp |

fclass linear probe: 79.65 / 86.56. kNN-Jaccard vs POI2Vec: 0.029 / 0.020.

## Verdict

⚠ Disqualified despite numeric dominance.

The +32 pp cat lift is a **leak artifact**: a linear probe on D's
last-step check-in embedding recovers the cat label at 51% (vs 31% for
canonical c2hgi). The bidirectional check-in↔POI edges plus 2-hop GCN
allow POI2Vec semantics from *future* visits in the same trajectory to
flow back into the current check-in's representation, leaking the target
category through the graph topology.

The reg lift (+3 pp) matches B/H/I/J/M and is consistent with legitimate
POI-stable signal aggregation, but fclass at 80-87% is *below* B/H's 98%,
suggesting the POI side of the heterograph is less semantically clean.

## Action

Skipped for FL. Either redesign the edge set to remove the check-in→POI
back-edge, or reframe D as a "trajectory completion" baseline rather than
a next-category predictor.
