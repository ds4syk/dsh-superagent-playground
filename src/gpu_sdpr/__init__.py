"""GPU-parallel SDPR layer (prototype).

Batched dual log-det barrier forward + Theorem-2 implicit differentiation,
targeting the SDP relaxation of the SDPR paper (Holmes et al., T-RO 2025).
"""
from .barrier import BarrierSolver
from .layer import SDPRLayer
from .p6 import build_p6_sdp, build_objective_matrix, objective_scale, scale_objective

__all__ = [
    "BarrierSolver",
    "SDPRLayer",
    "build_p6_sdp",
    "build_objective_matrix",
    "objective_scale",
    "scale_objective",
]
