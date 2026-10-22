"""Session-aligned walk-forward: folds are consecutive, separated by purge/embargo, and confined to their sessions."""
from __future__ import annotations

import dataclasses

import pandas as pd

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.strategies.momentum_strategy import MomentumStrategy
from quant_intelligence.validation.walk_forward import MIN_FOLD_TRADES, fold_sessions, walk_forward_sessions

DAYS = list(pd.bdate_range("2024-01-01", periods=60).date)


def test_folds_are_consecutive_and_disjoint():
    folds = fold_sessions(DAYS, 6, purge=0, embargo=0)
    assert len(folds) == 6 and sum(len(f) for f in folds) == 60
    flat = [d for f in folds for d in f]
    assert flat == sorted(flat) and len(flat) == len(set(flat))


def test_purge_trims_the_tail_and_embargo_the_head():
    folds = fold_sessions(DAYS, 4, purge=2, embargo=1)
    full = fold_sessions(DAYS, 4, purge=0, embargo=0)
    assert folds[0][-1] == full[0][-3]                       # last 2 sessions of the first fold purged
    assert folds[1][0] == full[1][1]                         # first session of the second fold embargoed
    assert folds[-1][-1] == full[-1][-1]                     # the final fold keeps its tail
    gap = DAYS.index(folds[1][0]) - DAYS.index(folds[0][-1]) - 1
    assert gap == 2 + 1


def test_too_few_sessions_gives_no_folds():
    assert fold_sessions(DAYS[:3], 6, 0, 0) == []


def test_each_fold_only_sees_its_own_sessions(synthetic_ohlcv, synthetic_features):
    sessions = sorted(pd.to_datetime(synthetic_ohlcv["timestamp"]).dt.date.unique())
    pol = dataclasses.replace(SETTINGS.splits, purge_sessions=1, embargo_sessions=1)
    wf = walk_forward_sessions(MomentumStrategy, synthetic_ohlcv, synthetic_features, "NIFTY", sessions, quantity=75, n_folds=4, policy=pol)
    assert 2 <= len(wf.folds) <= 4
    for a, b in zip(wf.folds, wf.folds[1:]):
        assert a.end < b.start                               # no shared session, and a gap between folds
    assert wf.pooled_trades == sum(f.n_trades for f in wf.folds)


def test_summary_statistics_follow_from_the_folds(synthetic_ohlcv, synthetic_features):
    sessions = sorted(pd.to_datetime(synthetic_ohlcv["timestamp"]).dt.date.unique())
    wf = walk_forward_sessions(MomentumStrategy, synthetic_ohlcv, synthetic_features, "NIFTY", sessions, quantity=75, n_folds=3)
    judged = [f.net_expectancy_r for f in wf.folds if f.n_trades >= MIN_FOLD_TRADES]
    assert wf.n_judged == len(judged)
    if judged:
        assert abs(wf.share_positive - sum(x > 0 for x in judged) / len(judged)) < 1e-12
        assert wf.worst_fold_r == min(judged)
    assert "folds" in wf.as_dict()


def test_thin_folds_do_not_vote_on_consistency(synthetic_ohlcv, synthetic_features):
    sessions = sorted(pd.to_datetime(synthetic_ohlcv["timestamp"]).dt.date.unique())
    wf = walk_forward_sessions(MomentumStrategy, synthetic_ohlcv, synthetic_features, "NIFTY", sessions, quantity=75, n_folds=12)
    assert wf.n_judged <= len([f for f in wf.folds if f.n_trades >= MIN_FOLD_TRADES])
