import json
import sys

from vllm_gaudi.platform import HpuPlatform

def _uses_lmcache_connector() -> bool:
    """Check if lmcache is configured as the KV connector via CLI args."""
    for i, arg in enumerate(sys.argv):
        if arg == "--kv-transfer-config" and i + 1 < len(sys.argv):
            try:
                config = json.loads(sys.argv[i + 1])
                connector = config.get("kv_connector", "")
                return "LMCache" in connector
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
