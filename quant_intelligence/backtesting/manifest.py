"""
Run manifest: everything needed to reproduce, audit and compare a backtest.

Given the same dataset fingerprint, strategy version, parameters, execution assumptions and engine version
the engine must produce the same trades (it uses no randomness; `random_seed` is recorded as None for that
reason, and any future stochastic step must record its seed here). `config_hash` covers every field that
affects the result and nothing that does not (no run timestamp, no run id), so two runs with the same
config_hash are expected to be identical and a differing hash explains why two results differ.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

from quant_intelligence.backtesting.timing import timing_override
from quant_intelligence.config.settings import PROJECT_ROOT, SETTINGS

ENGINE_VERSION = "2.0.0"  # bump when simulation, cost or metric logic changes in a way that can alter results


@dataclass
class RunManifest:
    data_source: str
    data_start: str | None
    data_end: str | None
    n_bars: int
    dataset_fingerprint: str
    instrument: str
    timeframe: str
    strategy: str
    strategy_version: str
    parameters: dict
    execution_timing: str
    quantity: int
    max_holding_bars: int
    cost_model: dict
    premium_model: str  # "bs" (Black-Scholes) or "fixed_pct" (assumed premium %)
    stop_atr_scale: float
    random_seed: int | None
    validation_method: str
    engine_version: str = ENGINE_VERSION
    code_commit: str = ""
    run_timestamp: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))
    config_hash: str = ""

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _canonical(value):
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


@lru_cache(maxsize=1)
def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(PROJECT_ROOT), capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def build_manifest(
    strategy,
    ohlcv: pd.DataFrame,
    instrument: str,
    split: str,
    quantity: int,
    max_holding_bars: int,
    timeframe: str = "5min",
    data_source: str = "",
    validation_method: str | None = None,
) -> RunManifest:
    from quant_intelligence.data.quality_gates import dataset_fingerprint

    n = len(ohlcv)
    manifest = RunManifest(
        data_source=data_source or "unspecified",
        data_start=str(ohlcv["timestamp"].min()) if n else None,
        data_end=str(ohlcv["timestamp"].max()) if n else None,
        n_bars=n,
        dataset_fingerprint=dataset_fingerprint(ohlcv, data_source, timeframe, instrument),
        instrument=instrument,
        timeframe=timeframe,
        strategy=strategy.name,
        strategy_version=strategy.version,
        parameters=dict(strategy.parameters),
        execution_timing=timing_override() or SETTINGS.execution_timing,
        quantity=quantity,
        max_holding_bars=max_holding_bars,
        cost_model=dataclasses.asdict(SETTINGS.costs),
        premium_model="bs" if SETTINGS.vol_premium_model_in_backtest else "fixed_pct",
        stop_atr_scale=SETTINGS.stop_atr_scale,
        random_seed=None,
        validation_method=validation_method or f"single chronological pass ({split})",
        code_commit=_git_commit(),
    )
    manifest.config_hash = hashlib.sha256(_canonical({
        k: v for k, v in manifest.as_dict().items() if k not in ("run_timestamp", "config_hash", "code_commit")
    }).encode()).hexdigest()
    return manifest
