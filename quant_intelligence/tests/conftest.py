import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import datetime as dt
import os
import tempfile

# Must happen before quant_intelligence.config.settings is imported: keeps tests from
# writing rows into the real dashboard database.
os.environ["LOGS_DIR"] = tempfile.mkdtemp(prefix="qi_test_logs_")  # tests must not write into the real logs/
os.environ["OPTION_UNDERLYINGS"] = "NIFTY,BANKNIFTY,FINNIFTY,MIDCPNIFTY,SENSEX"  # not whatever the developer put in .env
for _flag in ("VOL_MODELS_ENABLED", "VOL_FEATURES_IN_ANALOGUES", "VOL_PREMIUM_MODEL_IN_BACKTEST", "VOL_TARGET_SIZING",
              "VOL_RISK_CHECK", "VOL_IV_GATE", "OPTION_HISTORY_IN_BACKTEST", "CHAIN_RECORDER", "GATED_RANKER",
              "RESEARCH_SCHEDULER"):
    os.environ[_flag] = "false"  # a developer's .env must not change test results
os.environ["DHAN_CLIENT_ID"] = ""  # a developer's .env credentials/expired token must not leak into tests
os.environ["DHAN_ACCESS_TOKEN"] = ""
os.environ["EOD_REPORT"] = "false"  # no background reports or forced square-offs in tests
os.environ["MAX_RISK_PER_TRADE_PCT"] = "1.0"  # the pre-0.2% sizing the existing tests were written against
os.environ["MAX_DAILY_LOSS_PCT"] = "3.0"
os.environ["MAX_CAPITAL_PER_TRADE_PCT"] = "5.0"
os.environ["STOP_ATR_SCALE"] = "1.0"
os.environ["MOVER_UNIVERSE"] = "watchlist"  # tests never call NSE; the NSE path is tested with a fake loader
os.environ["MOVER_SELECTION_ENABLED"] = "false"  # tests scan everything unless they switch the selection on themselves
os.environ["TOKEN_KEEPER"] = "false"  # tests must never renew/replace tokens or call Dhan auth
os.environ["DATA_KEEPER"] = "false"  # the real background thread must never start (or call Dhan) during tests
os.environ["DATABASE_URL"] = "sqlite:///" + (Path(tempfile.mkdtemp(prefix="qi_test_db_")) / "test.db").as_posix()

import pytest

from quant_intelligence.data_adapters.synthetic import SyntheticAdapter
from quant_intelligence.features.feature_engine import compute_features


@pytest.fixture(scope="session", autouse=True)
def _test_database():
    from quant_intelligence.database.db import init_db

    init_db()


@pytest.fixture(autouse=True)
def _fresh_dhan_state():
    """Process-wide Dhan state (rate limiter, breaker, caches) must not leak between tests."""
    from quant_intelligence.brokers.dhan_api_client import clear_expiry_cache
    from quant_intelligence.brokers.dhan_cache import clear_dhan_caches
    from quant_intelligence.brokers.dhan_rate_limit import LIMITER

    clear_expiry_cache()
    clear_dhan_caches()
    LIMITER.reset()
    yield
    LIMITER.reset()
    clear_dhan_caches()


@pytest.fixture(scope="session")
def synthetic_ohlcv():
    adapter = SyntheticAdapter(seed=7)
    end = dt.datetime(2024, 6, 28, 15, 30)
    start = end - dt.timedelta(days=45)
    return adapter.get_ohlcv("NIFTY", "5min", start, end)


@pytest.fixture(scope="session")
def synthetic_features(synthetic_ohlcv):
    return compute_features(synthetic_ohlcv)
