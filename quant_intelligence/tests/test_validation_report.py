"""validate_backtest: from a real BacktestResult to an evidence state."""
from __future__ import annotations

import dataclasses
import json

from quant_intelligence.backtesting.engine import run_backtest
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.strategies.iv_crush_writer_strategy import IVCrushWriterStrategy
from quant_intelligence.strategies.momentum_strategy import MomentumStrategy
from quant_intelligence.strategies.orb_strategy import ORBStrategy
from quant_intelligence.validation.evidence import EvidenceState as S
from quant_intelligence.validation.report import validate_backtest


def test_short_history_is_insufficient_data_not_a_ranking(synthetic_ohlcv, synthetic_features):
    result = run_backtest(ORBStrategy(), synthetic_ohlcv, synthetic_features, "NIFTY", persist=False)
    report = validate_backtest(result, synthetic_ohlcv, quantity=75)
    assert report.evidence.state is S.INSUFFICIENT_DATA          # 45 days is far below the 120-session policy
    assert any("sessions" in r for r in report.evidence.reasons)
    assert report.config_hash == result.manifest["config_hash"] and report.run_id == result.run_id
    assert report.metrics.n_trades == result.metrics["n_trades"]


def test_metrics_agree_with_the_engine_own_summary(synthetic_ohlcv, synthetic_features):
    result = run_backtest(MomentumStrategy(), synthetic_ohlcv, synthetic_features, "NIFTY", persist=False, quantity=75)
    m = validate_backtest(result, synthetic_ohlcv, quantity=75).metrics
    assert m.n_trades == result.metrics["n_trades"] > 0
    assert abs(m.net_expectancy_r - result.metrics["net_expected_r"]) < 1e-9
    assert abs(m.net_pnl_total - result.metrics["net_pnl_total"]) < 1e-6
    assert abs(m.gross_pnl_total - result.metrics["gross_pnl_total"]) < 1e-6
    assert abs(m.max_drawdown_r - result.metrics["max_drawdown_r"]) < 1e-9


def test_chain_aware_strategy_is_never_judged_on_underlying_prices(synthetic_ohlcv, synthetic_features):
    result = run_backtest(IVCrushWriterStrategy(), synthetic_ohlcv, synthetic_features, "NIFTY", persist=False)
    report = validate_backtest(result, synthetic_ohlcv, quantity=75, policy=dataclasses.replace(SETTINGS.evidence, min_sessions=1))
    assert report.evidence.state is S.INSUFFICIENT_DATA and "option-chain" in report.evidence.reasons[0]


def test_relaxing_the_history_requirement_moves_on_to_the_sample_checks(synthetic_ohlcv, synthetic_features):
    result = run_backtest(ORBStrategy(), synthetic_ohlcv, synthetic_features, "NIFTY", persist=False)
    relaxed = dataclasses.replace(SETTINGS.evidence, min_sessions=10)
    report = validate_backtest(result, synthetic_ohlcv, quantity=75, policy=relaxed)
    assert report.evidence.state is S.INSUFFICIENT_SAMPLE         # ~40 trades on 30 sessions is still too thin


def test_report_serialises_to_json_with_no_nan_leaks(synthetic_ohlcv, synthetic_features):
    result = run_backtest(ORBStrategy(), synthetic_ohlcv, synthetic_features, "NIFTY", persist=False)
    text = json.dumps(validate_backtest(result, synthetic_ohlcv, quantity=75).as_dict(), default=str)
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["evidence"]["state"] == "INSUFFICIENT DATA"


def test_a_backtest_with_no_trades_is_handled(synthetic_ohlcv, synthetic_features):
    empty = synthetic_ohlcv.iloc[:6].reset_index(drop=True)
    result = run_backtest(ORBStrategy(), empty, synthetic_features.iloc[:6].reset_index(drop=True), "NIFTY", persist=False)
    report = validate_backtest(result, empty, quantity=75, data_quality="OK", n_sessions=500)
    assert report.metrics.n_trades == 0 and report.evidence.state is S.INSUFFICIENT_SAMPLE
