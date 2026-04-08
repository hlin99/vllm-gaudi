# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
import types
from pathlib import Path

import pytest


MODULE_NAME = "test_proxy_server_under_test"
STUBBED_MODULES = [
    MODULE_NAME,
    "aiohttp",
    "httpx",
    "colorlog",
    "colorlog.escape_codes",
    "fastapi",
    "fastapi.responses",
    "fastapi.middleware",
    "fastapi.middleware.cors",
    "transformers",
    "msgspec",
    "zmq",
    "zmq.asyncio",
    "uvicorn",
    "lmcache",
    "lmcache.v1",
    "lmcache.v1.storage_backend",
    "lmcache.v1.storage_backend.pd_backend",
]


def find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate repository root from test file path.")


def install_proxy_server_stubs() -> None:
    def ensure_module(name: str) -> types.ModuleType:
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            sys.modules[name] = module
        return module

    aiohttp = ensure_module("aiohttp")

    class ClientTimeout:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class ClientSession:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class ContentTypeError(Exception):
        pass

    class ClientError(Exception):
        pass

    aiohttp.ClientTimeout = ClientTimeout
    aiohttp.ClientSession = ClientSession
    aiohttp.ContentTypeError = ContentTypeError
    aiohttp.ClientError = ClientError

    httpx = ensure_module("httpx")

    class AsyncClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    httpx.AsyncClient = AsyncClient

    colorlog = ensure_module("colorlog")
    colorlog_escape_codes = ensure_module("colorlog.escape_codes")
    colorlog_escape_codes.escape_codes = {
        "cyan": "",
        "green": "",
        "yellow": "",
        "red": "",
        "reset": "",
    }
    colorlog.escape_codes = colorlog_escape_codes

    fastapi = ensure_module("fastapi")
    fastapi_responses = ensure_module("fastapi.responses")
    fastapi_middleware = ensure_module("fastapi.middleware")
    fastapi_cors = ensure_module("fastapi.middleware.cors")

    class APIRouter:
        def post(self, *args, **kwargs):
            return lambda func: func

        def get(self, *args, **kwargs):
            return lambda func: func

        def options(self, *args, **kwargs):
            return lambda func: func

    def Depends(value):
        return value

    def Header(*args, **kwargs):
        return kwargs.get("default")

    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class Request:
        pass

    class FastAPI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.state = types.SimpleNamespace()

        def add_middleware(self, *args, **kwargs):
            return None

        def include_router(self, *args, **kwargs):
            return None

    class JSONResponse:
        def __init__(self, content=None, status_code=200):
            self.content = content
            self.status_code = status_code

    class StreamingResponse:
        def __init__(self, content=None, media_type=None):
            self.content = content
            self.media_type = media_type

    class PlainTextResponse:
        def __init__(self, content=None, status_code=200):
            self.content = content
            self.status_code = status_code

    fastapi.APIRouter = APIRouter
    fastapi.Depends = Depends
    fastapi.FastAPI = FastAPI
    fastapi.Header = Header
    fastapi.HTTPException = HTTPException
    fastapi.Request = Request
    fastapi.status = types.SimpleNamespace(
        HTTP_500_INTERNAL_SERVER_ERROR=500,
        HTTP_403_FORBIDDEN=403,
    )
    fastapi_responses.JSONResponse = JSONResponse
    fastapi_responses.StreamingResponse = StreamingResponse
    fastapi_responses.PlainTextResponse = PlainTextResponse
    fastapi_cors.CORSMiddleware = object
    fastapi.middleware = fastapi_middleware

    transformers = ensure_module("transformers")

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, prompt):
            return {
                "input_ids": list(range(len(prompt))) if isinstance(prompt, str) else []
            }

    transformers.AutoTokenizer = AutoTokenizer

    msgspec = ensure_module("msgspec")

    class DecodeError(Exception):
        pass

    msgspec.DecodeError = DecodeError
    msgspec.msgpack = types.SimpleNamespace(decode=lambda *args, **kwargs: None)

    zmq = ensure_module("zmq")
    zmq_asyncio = ensure_module("zmq.asyncio")

    class ZMQError(Exception):
        def __init__(self, errno=None):
            super().__init__(errno)
            self.errno = errno

    class Context:
        def socket(self, *args, **kwargs):
            return object()

    zmq.PULL = 1
    zmq.Again = Exception
    zmq.ZMQError = ZMQError
    zmq.ETERM = 2
    zmq.ENOTSOCK = 3
    zmq.asyncio = zmq_asyncio
    zmq_asyncio.Context = Context

    uvicorn = ensure_module("uvicorn")
    uvicorn.Config = object
    uvicorn.Server = object

    lmcache = ensure_module("lmcache")
    lmcache_v1 = ensure_module("lmcache.v1")
    lmcache_storage_backend = ensure_module("lmcache.v1.storage_backend")
    lmcache_pd_backend = ensure_module("lmcache.v1.storage_backend.pd_backend")

    class PDMsg:
        pass

    class ProxyNotif(PDMsg):
        pass

    lmcache.v1 = lmcache_v1
    lmcache_v1.storage_backend = lmcache_storage_backend
    lmcache_storage_backend.pd_backend = lmcache_pd_backend
    lmcache_pd_backend.PDMsg = PDMsg
    lmcache_pd_backend.ProxyNotif = ProxyNotif


def load_proxy_server_module():
    install_proxy_server_stubs()
    proxy_server_path = find_repo_root() / "pd_xpyd" / "proxy_server.py"
    sys.modules.pop(MODULE_NAME, None)
    spec = importlib.util.spec_from_file_location(MODULE_NAME, proxy_server_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def restore_stubbed_modules():
    original_modules = {
        name: sys.modules.get(name)
        for name in STUBBED_MODULES
    }
    yield
    for name, module in original_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_calculate_message_token_length_uses_length_from_tokenizer_result():
    proxy_server = load_proxy_server_module()

    def fake_token_length_getter(content):
        return [content], len(content)

    total_length = proxy_server.calculate_message_token_length(
        [
            {"content": "hello"},
            {"content": "world!"},
        ],
        fake_token_length_getter,
    )

    assert total_length == 11


def test_calculate_message_token_length_requires_content_field():
    proxy_server = load_proxy_server_module()

    def fake_token_length_getter(content):
        return [content], len(content)

    try:
        proxy_server.calculate_message_token_length(
            [{"role": "user"}],
            fake_token_length_getter,
        )
    except ValueError as exc:
        assert str(exc) == "Each chat message must include a 'content' field."
    else:
        raise AssertionError("Expected a ValueError for messages without content.")


def test_resolve_decode_receiver_uses_configured_ports_and_offset_host():
    proxy_server = load_proxy_server_module()

    receiver_host, init_ports, alloc_ports = proxy_server.resolve_decode_receiver(
        "10.0.0.8:9305",
        decoder_init_port=[7310, 7311],
        decoder_alloc_port=[7410, 7411],
    )

    assert receiver_host == "10.0.0.13"
    assert init_ports == [7310, 7311]
    assert alloc_ports == [7410, 7411]


def test_build_disagg_spec_keeps_localhost_and_configured_ports():
    proxy_server = load_proxy_server_module()

    disagg_spec = proxy_server.build_disagg_spec(
        "42",
        "localhost:9400",
        decoder_init_port=[7501],
        decoder_alloc_port=[7601],
    )

    assert disagg_spec == {
        "req_id": "42",
        "receiver_host": "localhost",
        "receiver_init_port": [7501],
        "receiver_alloc_port": [7601],
    }


def test_resolve_decode_receiver_rejects_malformed_instance_strings():
    proxy_server = load_proxy_server_module()

    try:
        proxy_server.resolve_decode_receiver("missing-port")
    except ValueError as exc:
        assert str(exc) == (
            "Invalid decode instance specification. Expected format 'host:port'."
        )
    else:
        raise AssertionError("Expected a ValueError for malformed decode instances.")
