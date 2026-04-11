import os

# Enable CPU fallback for ops not yet implemented on MPS. This is a no-op
# on CUDA/CPU; on MPS it lets unsupported ops silently delegate to CPU
# instead of erroring out — important safety net for transformer kernels.
# `setdefault` respects any user-provided value.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import torch

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# DEVICE = torch.device("cpu")

CATEGORIES_MAP: dict[int, str] = {
    0: 'Community',
    1: 'Entertainment',
    2: 'Food',
    3: 'Nightlife',
    4: 'Outdoors',
    5: 'Shopping',
    6: 'Travel',
    7: 'None'
}
