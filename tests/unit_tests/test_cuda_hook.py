# SPDX-License-Identifier: Apache-2.0

import enum
import sys
from unittest.mock import MagicMock

import torch


# Provide minimal stubs for the vllm platform system so that
# HpuPlatform can be imported without real Gaudi / vllm packages.
class _PlatformEnum(enum.Enum):
    OOT = "OOT"


class _Platform:
    pass


_platforms_mock = MagicMock()
_platforms_mock.Platform = _Platform
_platforms_mock.PlatformEnum = _PlatformEnum

for mod in [
    "habana_frameworks",
    "habana_frameworks.torch",
    "habana_frameworks.torch.utils",
    "habana_frameworks.torch.utils.internal",
    "vllm",
    "vllm.envs",
    "vllm.config",
    "vllm.utils",
    "vllm.utils.torch_utils",
    "vllm_gaudi.extension",
    "vllm_gaudi.extension.runtime",
    "vllm_gaudi.extension.logger",
]:
    sys.modules.setdefault(mod, MagicMock())

sys.modules.setdefault("vllm.platforms", _platforms_mock)

from vllm_gaudi.platform import HpuPlatform  # noqa: E402


class TestCudaPostInit:

    def test_cuda_is_available_returns_false(self):
        """After cuda_post_init, torch.cuda.is_available() should
        always return False on HPU."""
        original = torch.cuda.is_available
        try:
            HpuPlatform.cuda_post_init()
            assert torch.cuda.is_available() is False
        finally:
            torch.cuda.is_available = original
