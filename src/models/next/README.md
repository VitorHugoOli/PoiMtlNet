# Next-Task Head Variants

This domain tracks head variants for the next-task branch.

Canonical runtime model registration remains in `src/models/next/head.py`.
Each variant folder documents why it exists and maps to the runtime class.

Variant folder contract:
- `README.md` using `docs/VARIANT_README_TEMPLATE.md`
- `metadata.yaml` with `evidence_status` in `{proposed, implemented, ablated, promoted}`
- `head.py` runtime entrypoint
