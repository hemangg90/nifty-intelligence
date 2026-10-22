"""Every backtest records what it ran on, and identical inputs give identical results."""
from __future__ import annotations

import dataclasses

import pytest

from quant_intelligence.backtesting import engine
from quant_intelligence.strategies.orb_strategy import ORBStrategy

REQUIRED = {
    "data_source", "data_start", "data_end", "n_bars", "dataset_fingerprint", "instrument", "timeframe", "strategy",
    "strategy_version", "parameters", "execution_timing", "quantity", "max_holding_bars", "cost_model", "premium_model",
    "random_seed", "validation_method", "run_timestamp", "engine_version", "config_hash",
}


def _run(ohlcv, feats, **kw):
    return engine.run_backtest(ORBStrategy(), ohlcv, feats, "NIFTY", persist=False, data_source="synthetic", **kw)


def _nan_safe(metrics):
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in metrics.items()}


def _signature(run):
    return [(t.setup.timestamp, t.setup.direction, t.setup.entry_price, t.exit_timestamp, t.exit_price, t.r_multiple) for t in run.trades]


def test_manifest_has_every_required_field(synthetic_ohlcv, synthetic_features):
    m = _run(synthetic_ohlcv, synthetic_features).manifest
    assert REQUIRED <= set(m)
    assert m["strategy"] == "Opening Range Breakout" and m["strategy_version"] == "1.0"
    assert m["data_source"] == "synthetic" and m["n_bars"] == len(synthetic_ohlcv) and len(m["dataset_fingerprint"]) == 64
    assert m["cost_model"]["brokerage_per_order_inr"] > 0 and m["parameters"]["stop_atr_mult"] == 0.75


def test_same_inputs_give_identical_trades_metrics_and_config_hash(synthetic_ohlcv, synthetic_features):
    a, b = _run(synthetic_ohlcv, synthetic_features), _run(synthetic_ohlcv, synthetic_features)
    assert _signature(a) == _signature(b)
    assert _nan_safe(a.metrics) == _nan_safe(b.metrics)
    assert a.manifest["config_hash"] == b.manifest["config_hash"]
    assert a.run_id != b.run_id  # ids are unique per run; the hash is what identifies the configuration


def test_changed_data_or_parameters_change_the_config_hash(synthetic_ohlcv, synthetic_features):
    base = _run(synthetic_ohlcv, synthetic_features).manifest
    shorter = _run(synthetic_ohlcv.iloc[:-50].reset_index(drop=True), synthetic_features.iloc[:-50].reset_index(drop=True)).manifest
    assert shorter["dataset_fingerprint"] != base["dataset_fingerprint"] and shorter["config_hash"] != base["config_hash"]
    other = engine.run_backtest(ORBStrategy({"stop_atr_mult": 1.0}), synthetic_ohlcv, synthetic_features, "NIFTY",
                                persist=False, data_source="synthetic").manifest
    assert other["config_hash"] != base["config_hash"]
    assert _run(synthetic_ohlcv, synthetic_features, quantity=75).manifest["config_hash"] != base["config_hash"]


def test_execution_timing_is_part_of_the_manifest(synthetic_ohlcv, synthetic_features, monkeypatch):
    from quant_intelligence.backtesting import manifest as manifest_mod
    from quant_intelligence.config import settings as settings_mod

    base = _run(synthetic_ohlcv, synthetic_features).manifest
    patched = dataclasses.replace(settings_mod.SETTINGS, execution_timing="next_open")
    monkeypatch.setattr(engine, "SETTINGS", patched)
    monkeypatch.setattr(manifest_mod, "SETTINGS", patched)
    fixed = _run(synthetic_ohlcv, synthetic_features).manifest
    assert base["execution_timing"] == "signal_close" and fixed["execution_timing"] == "next_open"
    assert fixed["config_hash"] != base["config_hash"]


def test_gross_and_net_are_reported_separately(synthetic_ohlcv, synthetic_features):
    m = _run(synthetic_ohlcv, synthetic_features).metrics
    if m.get("n_trades"):
        assert m["total_costs"] > 0
        assert m["gross_pnl_total"] == pytest.approx(m["net_pnl_total"] + m["total_costs"])


def test_manifest_is_persisted_with_the_run(synthetic_ohlcv, synthetic_features):
    from quant_intelligence.database.db import get_session
    from quant_intelligence.database.models import BacktestRun

    run = engine.run_backtest(ORBStrategy(), synthetic_ohlcv, synthetic_features, "NIFTY", persist=True, data_source="synthetic")
    with get_session() as s:
        row = s.query(BacktestRun).filter_by(run_id=run.run_id).one()
        assert row.manifest["config_hash"] == run.manifest["config_hash"] and row.timeframe == "5min"
