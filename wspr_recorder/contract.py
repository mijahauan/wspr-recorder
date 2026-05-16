"""HamSCI Client Contract v0.4 inventory and validate JSON builders."""

from __future__ import annotations

import logging
import os
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

from .config import Config, _derive_radiod_id
from .version import GIT_INFO

CONTRACT_VERSION = "0.4"
KA9Q_PYTHON_MIN_VERSION = "3.8.0"


def _instance_id(config: Config) -> str:
    return _derive_radiod_id(config.radiod.status_address)


def _chain_delay_ns(radiod_id: str) -> int | None:
    """Read RADIOD_<ID>_CHAIN_DELAY_NS from coordination.env (§8).

    Surfaced in inventory only — not applied to sample-to-UTC conversion
    since WSPR decoders are slot-quantized to minute boundaries, far
    outside the chain-delay regime. Present so a future tightening for
    sub-second drift work has the hook in place.
    """
    env_key = f"RADIOD_{radiod_id.upper().replace('-', '_')}_CHAIN_DELAY_NS"
    raw = os.environ.get(env_key, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _version_tuple(v: str) -> tuple[int, ...]:
    parts = []
    for p in v.split("."):
        m = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(m) if m else 0)
    return tuple(parts)


def _log_dir(config: Config) -> str:
    return os.environ.get("WSPR_RECORDER_LOG_DIR", "/var/log/wspr-recorder")


def build_inventory(config: Config, config_path: Path) -> dict:
    """Build the inventory --json payload per contract v0.4."""
    try:
        version = pkg_version("wspr-recorder")
    except Exception:
        version = "0.1.0"

    instance_id = _instance_id(config)
    log_dir = _log_dir(config)

    # Frequencies include both [[band]] entries and legacy [frequencies]
    band_freqs = sorted({bc.frequency for bc in config.bands})
    legacy_freqs = sorted(set(config.frequencies) - set(band_freqs))
    all_freqs = sorted(set(band_freqs) | set(legacy_freqs))

    modes_union: set[str] = set()
    for bc in config.bands:
        modes_union.update(bc.modes)

    instance = {
        "instance": instance_id,
        "radiod_id": instance_id,
        "host": "localhost",
        "radiod_status_dns": config.radiod.status_address,
        "data_destination": None,
        "ka9q_channels": len(all_freqs),
        "frequencies_hz": all_freqs,
        "modes": sorted(modes_union) if modes_union else ["W2"],
        "disk_writes": [
            {
                "path": config.recorder.output_dir,
                "mb_per_day": 0,
                "retention_days": 0,
            },
            {
                "path": log_dir,
                "mb_per_day": 5,
                "retention_days": 365,
            },
        ],
        "uses_timing_calibration": config.timing.authority in ("fusion", "auto"),
        "provides_timing_calibration": False,
        "chain_delay_ns_applied": _chain_delay_ns(instance_id),
    }

    # The process log goes to the systemd journal
    # (StandardOutput=journal) — see it via `smd log wspr-recorder`.
    # wspr-recorder writes no other file-based logs.
    log_paths: dict[str, Any] = {instance_id: {}}

    effective_level = logging.getLogger().getEffectiveLevel()
    log_level_name = logging.getLevelName(effective_level)

    payload: dict[str, Any] = {
        "client": "wspr-recorder",
        "version": version,
        "contract_version": CONTRACT_VERSION,
        "config_path": str(config_path),
    }
    if GIT_INFO:
        payload["git"] = GIT_INFO

    payload["log_paths"] = log_paths
    payload["log_level"] = log_level_name
    payload["instances"] = [instance]
    payload["deps"] = {
        "pypi": [{"name": "ka9q-python", "version": ">=3.8.0"}],
    }
    payload["issues"] = _collect_issues(config)
    return payload


def build_validate(config: Config, config_path: Path) -> dict:
    """Build the validate --json payload per contract v0.4.

    §12.3: report the absolute path of the loaded config.
    """
    issues = _collect_issues(config)
    return {
        "ok": not any(i["severity"] == "fail" for i in issues),
        "config_path": str(config_path),
        "issues": issues,
    }


def _collect_issues(config: Config) -> list[dict]:
    issues: list[dict] = []
    instance_id = _instance_id(config)

    # Reuse existing Config.validate() for structural checks, demote to warn/fail.
    for msg in config.validate():
        severity = "fail" if "must" in msg.lower() or "invalid" in msg.lower() or "no " in msg.lower() else "warn"
        issues.append({
            "severity": severity,
            "instance": instance_id,
            "message": msg,
        })

    if not (config.radiod.status_address or "").strip():
        issues.append({
            "severity": "fail",
            "instance": instance_id,
            "message": "radiod.status_address is empty",
        })

    # §12.2 SSRC uniqueness across (freq, preset, sample_rate, encoding).
    # In wspr-recorder all bands share channel_defaults, so duplicate
    # frequencies collapse to the same SSRC and MultiStream silently
    # drops the second registration.
    defaults = config.channel_defaults
    preset = getattr(defaults, "mode", "usb")
    sample_rate = getattr(defaults, "sample_rate", 12000)
    encoding = getattr(defaults, "encoding", "f32")

    seen: dict[tuple, int] = {}
    band_freqs = [bc.frequency for bc in config.bands]
    legacy_freqs = [f for f in config.frequencies if f not in band_freqs]
    for hz in band_freqs + legacy_freqs:
        key = (int(hz), preset, sample_rate, encoding)
        if key in seen:
            issues.append({
                "severity": "fail",
                "instance": instance_id,
                "message": (
                    f"SSRC collision: duplicate frequency {hz} Hz "
                    f"(preset={preset}, rate={sample_rate}, enc={encoding}) — "
                    f"ka9q-python MultiStream will silently drop one"
                ),
            })
        else:
            seen[key] = hz

    # §12.6 ka9q-python PyPI-lag check — warn when installed version is
    # older than the minimum declared in pyproject.toml.
    try:
        installed = pkg_version("ka9q-python")
    except Exception:
        installed = None
    if installed and _version_tuple(installed) < _version_tuple(KA9Q_PYTHON_MIN_VERSION):
        issues.append({
            "severity": "warn",
            "instance": instance_id,
            "message": (
                f"ka9q-python {installed} installed, "
                f"minimum required is {KA9Q_PYTHON_MIN_VERSION} — "
                f"run `pip install --upgrade ka9q-python`"
            ),
        })

    return issues
