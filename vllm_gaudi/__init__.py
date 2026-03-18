import json
import sys
import os

from vllm_gaudi.platform import HpuPlatform


def _uses_lmcache_connector() -> bool:
    """Check if lmcache is configured as the KV connector.

    Detection is based on:
    - Environment variables (for programmatic usage), and
    - CLI args (for command-line usage via --kv-transfer-config).
    """

    def _is_lmcache_connector(connector_value: str) -> bool:
        """Return True if the given connector string represents an LMCache connector."""
        if not isinstance(connector_value, str):
            return False
        return "LMCache" in connector_value

    # 1. Check env var that may mirror --kv-transfer-config JSON.
    #    This supports programmatic workflows that configure KVTransferConfig
    #    and then expose it via environment instead of CLI.
    env_kv_config = os.getenv("VLLM_KV_TRANSFER_CONFIG")
    if env_kv_config:
        try:
            config = json.loads(env_kv_config)
            connector = config.get("kv_connector", "")
            if _is_lmcache_connector(connector):
                return True
        except (json.JSONDecodeError, TypeError):
            # Fall through to other detection mechanisms.
            pass
    # 2. Check a simple env var that may directly specify the connector name.
    env_kv_connector = os.getenv("VLLM_KV_CONNECTOR")
    if env_kv_connector and _is_lmcache_connector(env_kv_connector):
        return True

    # 3. Fallback: inspect CLI args for --kv-transfer-config as before.
    for i, arg in enumerate(sys.argv):
        if arg == "--kv-transfer-config" and i + 1 < len(sys.argv):
            try:
                config = json.loads(sys.argv[i + 1])
                connector = config.get("kv_connector", "")
                return _is_lmcache_connector(connector)
            except (json.JSONDecodeError, TypeError):
                return False
    return False


def register():
    """Register the HPU platform."""
    HpuPlatform.set_torch_compile()
    if _uses_lmcache_connector():
        HpuPlatform.cuda_post_init()
    return "vllm_gaudi.platform.HpuPlatform"


def register_ops():
    """Register custom ops for the HPU platform."""
    import vllm_gaudi.v1.sample.hpu_rejection_sampler  # noqa: F401
    import vllm_gaudi.distributed.kv_transfer.kv_connector.v1.hpu_nixl_connector  # noqa: F401
    import vllm_gaudi.ops.hpu_fused_moe  # noqa: F401
    import vllm_gaudi.ops.hpu_layernorm  # noqa: F401
    import vllm_gaudi.ops.hpu_lora  # noqa: F401
    import vllm_gaudi.ops.hpu_rotary_embedding  # noqa: F401
    import vllm_gaudi.ops.hpu_compressed_tensors  # noqa: F401
    import vllm_gaudi.ops.hpu_fp8  # noqa: F401
    import vllm_gaudi.ops.hpu_gptq  # noqa: F401
    import vllm_gaudi.ops.hpu_awq  # noqa: F401
    import vllm_gaudi.ops.hpu_conv  # noqa: F401
    import vllm_gaudi.ops.hpu_mm_encoder_attention  # noqa: F401


def register_models():
    import vllm_gaudi.models.interfaces  # noqa: F401
    from .models import register_model
    register_model()
