"""SIMT GEMV kernels and their build-time catalogs (not Tensor Core decode)."""
from pathlib import Path


def measured_source(root, name):
    """Locate a body named by a pre-move receipt; never rewrite its digest."""
    path = Path(name)
    if path.parent == Path("quactlize/execution") and path.name not in (
        "api.h", "reader.hpp", "validation.hpp", "moe.h", "moe_prepare.cuh", "moe_router_warp.cuh"
    ):
        return root / "quactlize/simt_gemv" / path.name
    return root / path
