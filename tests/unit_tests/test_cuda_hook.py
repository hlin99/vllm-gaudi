# SPDX-License-Identifier: Apache-2.0

import enum
import sys
from unittest.mock import MagicMock, patch

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

import vllm_gaudi.utils  # noqa: E402
from vllm_gaudi.platform import HpuPlatform  # noqa: E402


class TestCudaPostInit:

    def test_real_hpu_returns_true(self):
        """On real HPU (not fake), torch.cuda.is_available() should
        return True after cuda_post_init."""
        original = torch.cuda.is_available
        try:
            with patch.object(vllm_gaudi.utils, "is_fake_hpu",
                              return_value=False):
                HpuPlatform.cuda_post_init()
                assert torch.cuda.is_available() is True
        finally:
            torch.cuda.is_available = original

    def test_fake_hpu_falls_back_to_original(self):
        """On fake HPU, torch.cuda.is_available() should fall back to
        the original torch implementation."""
        original = torch.cuda.is_available
        original_result = original()
        try:
            with patch.object(vllm_gaudi.utils, "is_fake_hpu",
                              return_value=True):
                HpuPlatform.cuda_post_init()
                assert torch.cuda.is_available() == original_result
        finally:
            torch.cuda.is_available = original
