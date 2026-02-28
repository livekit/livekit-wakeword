"""Stub for monotonic_align Cython extension.

The real implementation uses a Cython-compiled ``maximum_path_c`` function
that is only needed during VITS *training* (the ``forward()`` method of
``SynthesizerTrn``).  Our inference path never calls ``maximum_path``,
so this module only needs to exist for the import chain.
"""

import torch


def maximum_path(neg_cent: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    """Monotonic alignment search (stub — requires Cython extension for training)."""
    raise NotImplementedError("maximum_path requires the compiled Cython extension (training only)")
