"""MTL architecture domain."""

from .mtlnet.model import MTLnet
from .mtlnet_cgc.model import MTLnetCGC
from .mtlnet_crossattn.model import MTLnetCrossAttn
from .mtlnet_crossattn_dualtower.model import MTLnetCrossAttnDualTower
from .mtlnet_crossattn_dualtower_swiglu.model import MTLnetCrossAttnDualTowerSwiGLU
from .mtlnet_crossattn_swiglu.model import MTLnetCrossAttnSwiGLU
from .mtlnet_crossstitch.model import MTLnetCrossStitch
from .mtlnet_dselectk.model import MTLnetDSelectK
from .mtlnet_mmoe.model import MTLnetMMoE
from .mtlnet_ple.model import MTLnetPLE

__all__ = [
    "MTLnet", "MTLnetCGC", "MTLnetCrossAttn", "MTLnetCrossAttnDualTower",
    "MTLnetCrossAttnDualTowerSwiGLU", "MTLnetCrossAttnSwiGLU",
    "MTLnetCrossStitch", "MTLnetMMoE", "MTLnetDSelectK", "MTLnetPLE",
]
