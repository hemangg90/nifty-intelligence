"""Chronological session splits, purge/embargo, and the holdout ledger."""
from __future__ import annotations

import dataclasses
import uuid

import pandas as pd
import pytest

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.validation.splits import (
    SEGMENTS,
    holdout_fingerprint,
    holdout_status,
    record_holdout_evaluation,
    register_holdout,
    slice_segment,
    split_sessions,
)


def sessions_frame(n_days=100):
    days = pd.bdate_range("2024-01-01", periods=n_days)
    ts = [d + pd.Timedelta(hours=9, minutes=15 + 5 * i) for d in days for i in range(5)]
    return pd.DataFrame({"timestamp": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000})


def test_segments_are_chronological_and_never_overlap():
    s = split_sessions(sessions_frame())
    assert s.development[-1] < s.validation[0] and s.validation[-1] < s.holdout[0]
    all_days = s.development + s.validation + s.holdout
    assert len(all_days) == len(set(all_days))
    assert SEGMENTS == ("development", "validation", "holdout")


def test_purge_and_embargo_leave_a_gap_between_segments():
    s = split_sessions(sessions_frame(), dataclasses.replace(SETTINGS.splits, purge_sessions=2, embargo_sessions=3))
    business = list(pd.bdate_range("2024-01-01", periods=100).date)
    gap_dev_val = business.index(s.validation[0]) - business.index(s.development[-1]) - 1
    gap_val_hold = business.index(s.holdout[0]) - business.index(s.validation[-1]) - 1
    assert gap_dev_val == 2 + 3 and gap_val_hold == 2 + 3
    assert len(s.purged) == 4 and len(s.embargoed) == 6


def test_fractions_decide_the_segment_sizes():
    s = split_sessions(sessions_frame(100), dataclasses.replace(SETTINGS.splits, purge_sessions=0, embargo_sessions=0))
    assert (len(s.development), len(s.validation), len(s.holdout)) == (60, 20, 20)


def test_a_split_without_room_for_a_holdout_is_refused():
    with pytest.raises(ValueError):
        split_sessions(sessions_frame(), dataclasses.replace(SETTINGS.splits, dev_fraction=0.7, validation_fraction=0.3))
    with pytest.raises(ValueError):
        split_sessions(sessions_frame(2))


def test_a_segment_slice_contains_only_its_own_sessions():
    df = sessions_frame()
    feats = df[["timestamp"]].copy()
    s = split_sessions(df)
    o, f = slice_segment(df, feats, s, "validation")
    days = set(pd.to_datetime(o["timestamp"]).dt.date)
    assert days <= set(s.validation) | set(pd.bdate_range(s.validation[0], s.validation[-1]).date)
    assert len(o) == len(f) and (o["timestamp"].to_numpy() == f["timestamp"].to_numpy()).all()
    assert pd.to_datetime(o["timestamp"]).dt.date.min() == s.validation[0]


def test_holdout_fingerprint_changes_when_a_holdout_bar_changes_but_not_when_development_does():
    df = sessions_frame()
    s = split_sessions(df)
    base = holdout_fingerprint(df, df, s)
    dev_changed = df.copy()
    dev_changed.loc[0, "close"] = 555.0
    assert holdout_fingerprint(dev_changed, dev_changed, s) == base
    hold_changed = df.copy()
    hold_changed.loc[len(df) - 1, "close"] = 555.0
    assert holdout_fingerprint(hold_changed, hold_changed, s) != base


# ---- ledger --------------------------------------------------------------------------------------------------------

def fresh():
    df = sessions_frame()
    return split_sessions(df), "TEST_" + uuid.uuid4().hex[:8], "5min", "fp-" + uuid.uuid4().hex


def test_a_holdout_nobody_has_evaluated_is_untouched():
    s, inst, tf, fp = fresh()
    register_holdout(inst, tf, s, fp)
    register_holdout(inst, tf, s, fp)                       # idempotent
    st = holdout_status(inst, tf, fp)
    assert st.registered and st.evaluations == 0 and not st.consumed and st.first_evaluated_at is None


def test_evaluations_are_counted_and_distinct_strategies_tracked():
    s, inst, tf, fp = fresh()
    register_holdout(inst, tf, s, fp)
    record_holdout_evaluation(inst, tf, fp, "Momentum", "1.0", "cfgA", 40, 0.1, s)
    record_holdout_evaluation(inst, tf, fp, "Donchian", "1.0", "cfgB", 35, 0.0, s)
    st = holdout_status(inst, tf, fp)
    assert st.evaluations == 2 and st.distinct_strategies == 2 and st.repeat_looks == 0 and not st.consumed
    assert st.first_evaluated_at is not None


def test_looking_twice_at_the_same_configuration_consumes_the_holdout():
    s, inst, tf, fp = fresh()
    register_holdout(inst, tf, s, fp)
    for _ in range(2):
        record_holdout_evaluation(inst, tf, fp, "Momentum", "1.0", "cfgA", 40, 0.1, s)
    st = holdout_status(inst, tf, fp)
    assert st.consumed and st.repeat_looks == 1 and "more than once" in st.reason


def test_trying_too_many_strategies_consumes_the_holdout():
    s, inst, tf, fp = fresh()
    budget = dataclasses.replace(SETTINGS.splits, holdout_look_budget=2)
    for i in range(3):
        record_holdout_evaluation(inst, tf, fp, f"Strategy {i}", "1.0", f"cfg{i}", 30, 0.0, s)
    st = holdout_status(inst, tf, fp, budget)
    assert st.consumed and "different strategies" in st.reason


def test_new_data_means_a_fresh_untouched_holdout():
    s, inst, tf, fp = fresh()
    record_holdout_evaluation(inst, tf, fp, "Momentum", "1.0", "cfgA", 40, 0.1, s)
    record_holdout_evaluation(inst, tf, fp, "Momentum", "1.0", "cfgA", 40, 0.1, s)
    assert holdout_status(inst, tf, fp).consumed
    assert not holdout_status(inst, tf, "fp-new-data").consumed
