# mtlnet_cgc

Why this?
- CGC-lite introduces shared and task-specific experts while preserving the
  existing sequence-aware task encoders and heads.

Runtime mapping:
- Model registry key: `mtlnet_cgc`
- Runtime class: `models.mtlnet.MTLnetCGC`

Source:
- In-repo implementation: `src/models/mtlnet.py`
- Related literature: PLE/CGC (RecSys 2020)
