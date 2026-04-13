"""MTL architecture domain."""

from .mtlnet.model import MTLnet
from .mtlnet_cgc.model import MTLnetCGC
from .mtlnet_dselectk.model import MTLnetDSelectK
from .mtlnet_mmoe.model import MTLnetMMoE
from .mtlnet_ple.model import MTLnetPLE

__all__ = ["MTLnet", "MTLnetCGC", "MTLnetMMoE", "MTLnetDSelectK", "MTLnetPLE"]
