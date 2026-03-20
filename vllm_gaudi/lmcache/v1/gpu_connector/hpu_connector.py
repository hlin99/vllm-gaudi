# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
HPU (Habana/Gaudi) connector for LMCache GPU connector interface.

This module provides an HPU-specific implementation of the LMCache
GPU connector, adapting the KV cache transfer operations for Intel
Gaudi accelerators. It follows the same pattern as the XPU connector
in the upstream LMCache package.

Key HPU synchronization concepts:
    HPU operates in lazy execution mode by default, where operations are
    accumulated into a computational graph rather than executed immediately.
    ``htorch.core.mark_step()`` triggers the execution of all accumulated
    operations, serving as a synchronization point similar to how
    ``event.record()`` works on CUDA streams.

    On CUDA, ``event.record()`` records an event on a stream to track
    when preceding operations complete. On HPU, ``htorch.core.mark_step()``
    fulfills a similar role: it ensures that all preceding lazy operations
    (such as KV cache transfers via ``index_copy_`` / ``index_select`` with
    ``slot_mapping`` slicing) are submitted for execution on the device.
    This is essential for coordinating with vLLM's execution pipeline,
    which expects the KV cache to be populated before the model's forward
    pass consumes it.

    Coordination with vLLM:
        In vLLM's ``HPUModelRunner``, ``htorch.core.mark_step()`` is called
        at key boundaries (e.g., after batch preparation, before/after model
        forward, before/after sampling) to ensure lazy ops are flushed.
        Similarly, in this connector, ``mark_step()`` is called after KV cache
        transfer operations to guarantee that data is available in the KV cache
        before vLLM's attention layers read from it. Without this, the lazy
        execution engine may defer the ``index_copy_`` operations, leading to
        stale or missing KV cache data during the forward pass.
"""

# Standard
from typing import List, Optional

# Third Party
import torch

import habana_frameworks.torch as htorch

# First Party
from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata

from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()


class VLLMPagedMemHPUConnectorV2(VLLMPagedMemGPUConnectorV2):
    """
    HPU-specific connector for LMCache KV cache transfer operations.

    The GPU KV cache should be a nested tuple of K and V tensors.
    More specifically, we have:
    - GPUTensor = Tuple[KVLayer, ...]
    - KVLayer = Tuple[Tensor, Tensor]
    - Tensor: [num_blocks, block_size, num_heads, head_size]

    It will produce / consume memory object with KV_2LTD format.

    Unlike the CUDA connector which uses custom CUDA kernels and stream-based
    synchronization, this HPU connector uses PyTorch indexing operations
    (``index_copy_``, ``index_select``) and HPU's lazy execution model with
    ``htorch.core.mark_step()`` for synchronization.
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        """
        If use_gpu is true, it will create a gpu intermediate buffer. In this
        case, it requires the following kwargs:
        - chunk_size: The MAX size of the chunk to be copied to GPU.
        - dtype: The data type of the intermediate buffer.
        """
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.kv_cache_pointers = torch.empty(
            num_layers, dtype=torch.int64, device="cpu"
        )
        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0

        self.kvcaches: Optional[List[torch.Tensor]] = None
        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]
        if use_gpu:
            assert "chunk_size" in kwargs, (
                "chunk_size should be provided to create a GPU buffer."
            )
            assert "dtype" in kwargs, (
                "dtype should be provided to create a GPU buffer."
            )
            assert "device" in kwargs, (
                "device should be provided to create a GPU buffer."
            )
            shape = self.get_shape(kwargs["chunk_size"])
            self.gpu_buffer = torch.empty(
                shape, dtype=kwargs["dtype"], device=kwargs["device"]
            )

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMPagedMemHPUConnectorV2":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model
                configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.

        Returns:
            A new instance of VLLMPagedMemHPUConnectorV2.
        """
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        chunk_size = metadata.kv_shape[2]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
        )

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Copy KV cache data from a memory object into the HPU KV cache.

        This transfers data from host/CPU memory into the device KV cache
        using ``slot_mapping`` to determine the target positions.

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where its length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will start
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching region.

        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemHPUConnector"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    " order to be processed by VLLMPagedMemHPUConnector"
                )

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        slices = slot_mapping[start:end]

        if self.use_mla:
            tmp = memory_obj.tensor[0].to(slot_mapping.device)
            num_blocks, block_size, head_size = kvcaches[0].shape
            total_blocks = num_blocks * block_size
            for i, kvcache in enumerate(kvcaches):
                kvcache.view(total_blocks, head_size).index_copy_(
                    0, slices, tmp[i]
                )
        else:
            tmp_k = memory_obj.tensor[0].to(slot_mapping.device)
            tmp_v = memory_obj.tensor[1].to(slot_mapping.device)
            num_blocks, block_size, num_heads, head_size = \
                kvcaches[0][0].shape
            total_blocks = num_blocks * block_size
            d = num_heads * head_size
            for i, (kcache, vcache) in enumerate(kvcaches):
                kcache.view(total_blocks, d).index_copy_(
                    0, slices, tmp_k[i]
                )
                vcache.view(total_blocks, d).index_copy_(
                    0, slices, tmp_v[i]
                )

        # On HPU, mark_step() triggers execution of all accumulated lazy
        # operations (the index_copy_ calls above). This is the HPU
        # equivalent of CUDA's event.record() — it ensures the KV cache
        # writes are submitted to the device before vLLM's attention layers
        # attempt to read from the cache during the forward pass.
        htorch.core.mark_step()

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Copy KV cache data from the HPU KV cache into a memory object.

        This transfers data from the device KV cache into host/CPU memory
        using ``slot_mapping`` to determine the source positions.

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where its length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will start
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching region.

        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        slices = slot_mapping[start:end]

        if self.use_mla:
            num_blocks, block_size, head_size = kvcaches[0].shape
            total_blocks = num_blocks * block_size
            tmp = torch.stack(
                [
                    kvcache.view(total_blocks, head_size).index_select(
                        0, slices
                    )
                    for kvcache in kvcaches
                ]
            )
        else:
            num_blocks, block_size, num_heads, head_size = \
                kvcaches[0][0].shape
            total_blocks = num_blocks * block_size
            d = num_heads * head_size
            tmp_k = torch.stack(
                [
                    kvcache[0].view(total_blocks, d).index_select(0, slices)
                    for kvcache in kvcaches
                ]
            )
            tmp_v = torch.stack(
                [
                    kvcache[1].view(total_blocks, d).index_select(0, slices)
                    for kvcache in kvcaches
                ]
            )
            tmp = torch.stack([tmp_k, tmp_v])
        memory_obj.tensor.copy_(tmp, non_blocking=True)

        # On HPU, mark_step() flushes the lazy execution graph, ensuring
        # that the index_select and copy_ operations above are actually
        # submitted for execution. This is analogous to CUDA's
        # event.record() followed by synchronize() — it creates a
        # synchronization point so that the memory object's tensor
        # contains valid data before it is consumed downstream
        # (e.g., sent to the LMCache server or CPU backend).
        htorch.core.mark_step()

        if not memory_obj.tensor.device.type == "hpu":
            # Force a synchronize if the target buffer is NOT on HPU device.
            # This ensures the device-to-host transfer completes before the
            # data is accessed on the CPU side.
            torch.hpu.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    # TODO: need to optimize to enable real batching
    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends):
            self.to_gpu(memory_obj, start, end, **kwargs)

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends):
            self.from_gpu(memory_obj, start, end, **kwargs)
