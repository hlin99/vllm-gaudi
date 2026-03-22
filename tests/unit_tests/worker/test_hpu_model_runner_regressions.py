import ast
from pathlib import Path


def _get_model_runner_path() -> Path:
    for path in Path(__file__).resolve().parents:
        candidate = path / "vllm_gaudi" / "v1" / "worker" / "hpu_model_runner.py"
        if candidate.exists():
            return candidate
    raise AssertionError("Could not locate vllm_gaudi/v1/worker/hpu_model_runner.py")


def _get_method(tree: ast.AST, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"Could not find {class_name}.{method_name}")


def _parse_model_runner() -> ast.Module:
    return ast.parse(_get_model_runner_path().read_text())


def test_prepare_prefill_inputs_does_not_early_return_none():
    tree = _parse_model_runner()
    method = _get_method(tree, "HPUModelRunner", "_prepare_prefill_inputs")
    returns_none_tuple = [
        node for node in ast.walk(method)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Tuple)
        and len(node.value.elts) == 2
        and all(isinstance(elt, ast.Constant) and elt.value is None for elt in node.value.elts)
    ]
    assert not returns_none_tuple


def test_prepare_decode_inputs_does_not_batch_prefills_into_decode():
    tree = _parse_model_runner()
    method = _get_method(tree, "HPUModelRunner", "_prepare_decode_inputs")

    arg_names = [arg.arg for arg in method.args.args]
    assert arg_names == ["self", "num_decodes", "num_scheduled_tokens", "scheduler_output"]

    calls_create_prefix_prefill_decode = [
        node for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_create_decode_input_data_prefix_prefill"
    ]
    assert not calls_create_prefix_prefill_decode
