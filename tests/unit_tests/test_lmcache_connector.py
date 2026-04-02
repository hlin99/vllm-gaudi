# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2024-2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _ensure_vllm_gaudi_importable():
    """Set up mock modules so vllm_gaudi can be imported without HPU hardware.

    habana_frameworks, vllm (and some sub-modules) are stubbed out when they
    are not already available.  This only validates logic paths (env var /
    CLI detection, register() branching, cuda hook removal) and does not
    exercise actual HPU or vLLM integration behaviour.
    """

    for mod_name in [
            "habana_frameworks",
            "habana_frameworks.torch",
            "vllm",
            "vllm.envs",
            "vllm.platforms",
            "vllm_gaudi.extension",
            "vllm_gaudi.extension.runtime",
            "vllm_gaudi.extension.logger",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    # Provide the minimal symbols that platform.py expects at import time.
    vllm_platforms = sys.modules["vllm.platforms"]
    if not hasattr(vllm_platforms, "Platform"):
        vllm_platforms.Platform = type("Platform", (), {})  # type: ignore[attr-defined]
    if not hasattr(vllm_platforms, "PlatformEnum"):
        vllm_platforms.PlatformEnum = MagicMock()  # type: ignore[attr-defined]

    vllm_envs = sys.modules["vllm.envs"]
    if not hasattr(vllm_envs, "VLLM_USE_V1"):
        vllm_envs.VLLM_USE_V1 = False  # type: ignore[attr-defined]

    ext_runtime = sys.modules["vllm_gaudi.extension.runtime"]
    if not hasattr(ext_runtime, "get_config"):
        ext_runtime.get_config = MagicMock()  # type: ignore[attr-defined]

    ext_logger = sys.modules["vllm_gaudi.extension.logger"]
    if not hasattr(ext_logger, "logger"):
        ext_logger.logger = MagicMock()  # type: ignore[attr-defined]


# Ensure the module is importable *before* any test is collected.
_ensure_vllm_gaudi_importable()

# Now we can safely import from vllm_gaudi.
from vllm_gaudi import _uses_lmcache_connector  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_lmcache_env(monkeypatch):
    """Remove LMCache-related env vars before each test so tests are isolated."""
    monkeypatch.delenv("VLLM_KV_TRANSFER_CONFIG", raising=False)
    monkeypatch.delenv("VLLM_KV_CONNECTOR", raising=False)


# ---------------------------------------------------------------------------
# Tests for VLLM_KV_TRANSFER_CONFIG env var detection
# ---------------------------------------------------------------------------


class TestUsesLmcacheConnectorEnvConfig:
    """Tests for _uses_lmcache_connector() via VLLM_KV_TRANSFER_CONFIG env var."""

    def test_env_kv_config_with_lmcache(self, monkeypatch):
        config = json.dumps({"kv_connector": "LMCacheConnector"})
        monkeypatch.setenv("VLLM_KV_TRANSFER_CONFIG", config)
        assert _uses_lmcache_connector() is True

    def test_env_kv_config_with_lmcache_substring(self, monkeypatch):
        config = json.dumps({"kv_connector": "MyLMCacheConnector"})
        monkeypatch.setenv("VLLM_KV_TRANSFER_CONFIG", config)
        assert _uses_lmcache_connector() is True

    def test_env_kv_config_non_lmcache(self, monkeypatch):
        config = json.dumps({"kv_connector": "SimpleConnector"})
        monkeypatch.setenv("VLLM_KV_TRANSFER_CONFIG", config)
        assert _uses_lmcache_connector() is False

    def test_env_kv_config_missing_kv_connector_key(self, monkeypatch):
        config = json.dumps({"other_key": "value"})
        monkeypatch.setenv("VLLM_KV_TRANSFER_CONFIG", config)
        assert _uses_lmcache_connector() is False

    def test_env_kv_config_invalid_json(self, monkeypatch):
        monkeypatch.setenv("VLLM_KV_TRANSFER_CONFIG", "not-json")
        assert _uses_lmcache_connector() is False

    def test_env_kv_config_empty_string(self, monkeypatch):
        monkeypatch.setenv("VLLM_KV_TRANSFER_CONFIG", "")
        assert _uses_lmcache_connector() is False


# ---------------------------------------------------------------------------
# Tests for VLLM_KV_CONNECTOR env var detection
# ---------------------------------------------------------------------------


class TestUsesLmcacheConnectorEnvDirect:
    """Tests for _uses_lmcache_connector() via VLLM_KV_CONNECTOR env var."""

    def test_env_kv_connector_lmcache(self, monkeypatch):
        monkeypatch.setenv("VLLM_KV_CONNECTOR", "LMCacheConnector")
        assert _uses_lmcache_connector() is True

    def test_env_kv_connector_non_lmcache(self, monkeypatch):
        monkeypatch.setenv("VLLM_KV_CONNECTOR", "OtherConnector")
        assert _uses_lmcache_connector() is False

    def test_env_kv_connector_empty(self, monkeypatch):
        monkeypatch.setenv("VLLM_KV_CONNECTOR", "")
        assert _uses_lmcache_connector() is False


# ---------------------------------------------------------------------------
# Tests for CLI --kv-transfer-config detection
# ---------------------------------------------------------------------------


class TestUsesLmcacheConnectorCLI:
    """Tests for _uses_lmcache_connector() via CLI --kv-transfer-config args."""

    def test_cli_arg_with_lmcache(self, monkeypatch):
        config = json.dumps({"kv_connector": "LMCacheConnector"})
        monkeypatch.setattr(sys, "argv", ["vllm", "serve", "--kv-transfer-config", config])
        assert _uses_lmcache_connector() is True

    def test_cli_arg_non_lmcache(self, monkeypatch):
        config = json.dumps({"kv_connector": "SimpleConnector"})
        monkeypatch.setattr(sys, "argv", ["vllm", "serve", "--kv-transfer-config", config])
        assert _uses_lmcache_connector() is False

    def test_cli_arg_invalid_json(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["vllm", "serve", "--kv-transfer-config", "{bad-json}"])
        assert _uses_lmcache_connector() is False

    def test_cli_arg_missing_value(self, monkeypatch):
        """--kv-transfer-config is the last arg with no value following it."""
        monkeypatch.setattr(sys, "argv", ["vllm", "serve", "--kv-transfer-config"])
        assert _uses_lmcache_connector() is False


# ---------------------------------------------------------------------------
# Tests for no configuration at all
# ---------------------------------------------------------------------------


class TestUsesLmcacheConnectorNoConfig:
    """Tests for _uses_lmcache_connector() with no LMCache configuration."""

    def test_no_env_no_cli(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["vllm", "serve"])
        assert _uses_lmcache_connector() is False


# ---------------------------------------------------------------------------
# Tests for detection priority
# ---------------------------------------------------------------------------


class TestUsesLmcacheConnectorPriority:
    """Tests for detection priority: env config > env connector > CLI args."""

    def test_env_config_takes_priority_over_cli(self, monkeypatch):
        env_config = json.dumps({"kv_connector": "LMCacheConnector"})
        cli_config = json.dumps({"kv_connector": "SimpleConnector"})
        monkeypatch.setenv("VLLM_KV_TRANSFER_CONFIG", env_config)
        monkeypatch.setattr(sys, "argv", ["vllm", "serve", "--kv-transfer-config", cli_config])
        assert _uses_lmcache_connector() is True

    def test_env_connector_takes_priority_over_cli(self, monkeypatch):
        cli_config = json.dumps({"kv_connector": "SimpleConnector"})
        monkeypatch.setenv("VLLM_KV_CONNECTOR", "LMCacheConnector")
        monkeypatch.setattr(sys, "argv", ["vllm", "serve", "--kv-transfer-config", cli_config])
        assert _uses_lmcache_connector() is True


# ---------------------------------------------------------------------------
# Tests for HpuPlatform.remove_cuda_hooks()
# ---------------------------------------------------------------------------


class TestRemoveCudaHooks:
    """Tests for HpuPlatform.remove_cuda_hooks()."""

    def test_remove_cuda_hooks_disables_cuda(self):
        import torch
        from vllm_gaudi.platform import HpuPlatform
        with patch.object(torch.cuda, "is_available", torch.cuda.is_available):
            HpuPlatform.remove_cuda_hooks()
            assert torch.cuda.is_available() is False


# ---------------------------------------------------------------------------
# Tests for register() integration with LMCache detection
# ---------------------------------------------------------------------------


class TestRegisterWithLmcache:
    """Tests for register() integration with LMCache detection."""

    def test_register_calls_remove_cuda_hooks_when_lmcache(self):
        with patch("vllm_gaudi._uses_lmcache_connector", return_value=True), \
             patch("vllm_gaudi.HpuPlatform.set_torch_compile"), \
             patch("vllm_gaudi.HpuPlatform.remove_cuda_hooks") as mock_remove:
            from vllm_gaudi import register
            result = register()
            mock_remove.assert_called_once()
            assert result == "vllm_gaudi.platform.HpuPlatform"

    def test_register_skips_remove_cuda_hooks_when_no_lmcache(self):
        with patch("vllm_gaudi._uses_lmcache_connector", return_value=False), \
             patch("vllm_gaudi.HpuPlatform.set_torch_compile"), \
             patch("vllm_gaudi.HpuPlatform.remove_cuda_hooks") as mock_remove:
            from vllm_gaudi import register
            result = register()
            mock_remove.assert_not_called()
            assert result == "vllm_gaudi.platform.HpuPlatform"
