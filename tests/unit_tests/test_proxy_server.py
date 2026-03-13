# SPDX-License-Identifier: Apache-2.0
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional


PROXY_SERVER_PATH = Path(
    "/home/runner/work/vllm-gaudi/vllm-gaudi/pd_xpyd/proxy_server.py"
)
PROXY_SERVER_AST = ast.parse(PROXY_SERVER_PATH.read_text())
FUNCTIONS = {
    node.name: node
    for node in PROXY_SERVER_AST.body
    if isinstance(node, ast.FunctionDef)
}


def load_functions(*names: str, extra_globals: Optional[dict] = None) -> dict:
    namespace = {
        "Callable": Callable,
        "Optional": Optional,
    }
    if extra_globals:
        namespace.update(extra_globals)

    for name in names:
        module = ast.Module(body=[copy.deepcopy(FUNCTIONS[name])], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(PROXY_SERVER_PATH), "exec"), namespace)

    return namespace


def test_calculate_message_token_length_uses_length_from_tokenizer_result():
    namespace = load_functions("calculate_message_token_length")

    def fake_token_length_getter(content):
        return [content], len(content)

    total_length = namespace["calculate_message_token_length"](
        [
            {"content": "hello"},
            {"content": "world!"},
        ],
        fake_token_length_getter,
    )

    assert total_length == 11


def test_resolve_decode_receiver_uses_configured_ports_and_offset_host():
    namespace = load_functions(
        "resolve_decode_receiver",
        extra_globals={
            "global_args": SimpleNamespace(
                decoder_init_port=[7310, 7311],
                decoder_alloc_port=[7410, 7411],
            )
        },
    )

    receiver_host, init_ports, alloc_ports = namespace["resolve_decode_receiver"](
        "10.0.0.8:9305"
    )

    assert receiver_host == "10.0.0.13"
    assert init_ports == [7310, 7311]
    assert alloc_ports == [7410, 7411]


def test_build_disagg_spec_keeps_localhost_and_configured_ports():
    namespace = load_functions(
        "resolve_decode_receiver",
        "build_disagg_spec",
        extra_globals={
            "global_args": SimpleNamespace(
                decoder_init_port=[7501],
                decoder_alloc_port=[7601],
            )
        },
    )

    disagg_spec = namespace["build_disagg_spec"]("42", "localhost:9400")

    assert disagg_spec == {
        "req_id": "42",
        "receiver_host": "localhost",
        "receiver_init_port": [7501],
        "receiver_alloc_port": [7601],
    }
