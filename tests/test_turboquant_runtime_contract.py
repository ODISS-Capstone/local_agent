"""Contract gates for local_agent TurboQuant runtime integration."""
from __future__ import annotations

import builtins
import sys
import types
from typing import Any

from src.runtime import turboquant_runtime as runtime


ENV_KEYS = (
    "TURBOQUANT_AUTO_WRAP",
    "TURBOQUANT_KEY_BITS",
    "TURBOQUANT_VALUE_BITS",
    "TURBOQUANT_COMPRESS_VALUES",
    "TURBOQUANT_REQUIRE_CUDA",
)


def test_sync_env_applies_agent_config_defaults(monkeypatch) -> None:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    cfg = runtime._sync_env(  # noqa: SLF001 - intentional contract test
        {
            "auto_wrap": True,
            "key_bits": 4,
            "value_bits": 2,
            "compress_values": True,
            "require_cuda": False,
        }
    )

    assert cfg["key_bits"] == 4
    assert cfg["value_bits"] == 2
    assert cfg["compress_values"] is True
    assert cfg["require_cuda"] is False

    assert runtime.os.environ["TURBOQUANT_AUTO_WRAP"] == "1"
    assert runtime.os.environ["TURBOQUANT_KEY_BITS"] == "4"
    assert runtime.os.environ["TURBOQUANT_VALUE_BITS"] == "2"
    assert runtime.os.environ["TURBOQUANT_COMPRESS_VALUES"] == "1"
    assert runtime.os.environ["TURBOQUANT_REQUIRE_CUDA"] == "0"


def test_install_respects_auto_wrap_disable(monkeypatch) -> None:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    assert runtime.install({"auto_wrap": False}) is False


def test_install_uses_turboquant_runtime_hook(monkeypatch) -> None:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    fake_runtime_module = types.ModuleType("turboquant.runtime")
    fake_runtime_module.install_hf_autowrap = lambda force=True: True  # type: ignore[attr-defined]
    fake_pkg = types.ModuleType("turboquant")
    fake_pkg.runtime = fake_runtime_module  # type: ignore[attr-defined]

    # Ensure "from turboquant.runtime import install_hf_autowrap" resolves.
    monkeypatch.setitem(sys.modules, "turboquant", fake_pkg)
    monkeypatch.setitem(sys.modules, "turboquant.runtime", fake_runtime_module)

    assert runtime.install({"auto_wrap": True, "key_bits": 3}) is True


def test_wrap_uses_auto_wrap_when_available(monkeypatch) -> None:
    marker = object()

    fake_runtime_module = types.ModuleType("turboquant.runtime")
    fake_runtime_module.auto_wrap = lambda model: marker  # type: ignore[attr-defined]
    fake_pkg = types.ModuleType("turboquant")
    fake_pkg.runtime = fake_runtime_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "turboquant", fake_pkg)
    monkeypatch.setitem(sys.modules, "turboquant.runtime", fake_runtime_module)

    wrapped = runtime.wrap(object())
    assert wrapped is marker


def test_wrap_passthrough_without_turboquant(monkeypatch) -> None:
    model: Any = {"model": "dummy"}
    original_import = builtins.__import__

    def _raise_import_error(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001, ANN202
        if name == "turboquant.runtime":
            raise ImportError("missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", _raise_import_error)
    assert runtime.wrap(model) is model
