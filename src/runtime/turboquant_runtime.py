"""TurboQuant runtime integration for local_agent.

Ensures every HuggingFace Transformers model loaded inside the edge
agent is wrapped with the TurboQuant compressed KV cache by default.
All knobs are read from ``config/agent_config.yaml`` under the
``turboquant:`` key or the environment.

The helpers are defensive on CPU-only dev machines: when CUDA or the
TurboQuantWrapper package are missing they log a warning and return the
original model instead of raising.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping

logger = logging.getLogger(__name__)


DEFAULTS: dict[str, Any] = {
    "auto_wrap": True,
    "key_bits": 3,
    "value_bits": 3,
    "compress_values": False,
    "require_cuda": False,
}


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _sync_env(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    if config:
        for key in DEFAULTS:
            if key in config:
                cfg[key] = config[key]

    os.environ.setdefault(
        "TURBOQUANT_AUTO_WRAP", "1" if _bool(cfg["auto_wrap"], True) else "0"
    )
    os.environ.setdefault("TURBOQUANT_KEY_BITS", str(int(cfg["key_bits"])))
    os.environ.setdefault("TURBOQUANT_VALUE_BITS", str(int(cfg["value_bits"])))
    os.environ.setdefault(
        "TURBOQUANT_COMPRESS_VALUES", "1" if _bool(cfg["compress_values"], False) else "0"
    )
    os.environ.setdefault(
        "TURBOQUANT_REQUIRE_CUDA", "1" if _bool(cfg["require_cuda"], False) else "0"
    )
    return cfg


def install(config: Mapping[str, Any] | None = None) -> bool:
    """Activate TurboQuant auto-wrap for Transformers loads in this process."""
    cfg = _sync_env(config)
    if not _bool(cfg["auto_wrap"], True):
        logger.info("TurboQuant auto-wrap disabled by local_agent config")
        return False

    try:
        from turboquant.runtime import install_hf_autowrap
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "TurboQuantWrapper not importable on edge, skipping auto-wrap: %s", exc
        )
        return False

    installed = install_hf_autowrap(force=True)
    if installed:
        logger.info(
            "TurboQuant auto-wrap active on edge agent "
            "(key_bits=%s, compress_values=%s)",
            cfg["key_bits"], cfg["compress_values"],
        )
    else:
        logger.warning("TurboQuant auto-wrap requested but hook install failed")
    return installed


def wrap(model: Any) -> Any:
    """Explicit wrapper for loaders that bypass ``AutoModel*``."""
    try:
        from turboquant.runtime import auto_wrap
    except Exception as exc:  # noqa: BLE001
        logger.debug("TurboQuantWrapper unavailable on edge: %s", exc)
        return model
    return auto_wrap(model)


__all__ = ["install", "wrap"]
