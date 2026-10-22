"""
Thin REST client for the Dhan v2 API.

Centralizes auth headers, base URL, rate limiting and error handling for every Dhan endpoint the
system uses. Every request is assigned a Dhan API category (quote / data / non_trading / orders /
option_chain) and goes through the process-wide limiter in `dhan_rate_limit.LIMITER`, which spaces
calls below Dhan's account-wide limits. Read calls that get HTTP 429 back off and retry; after
repeated 429s a short circuit breaker fast-fails reads (never orders). Order placement is throttled
but NEVER auto-retried (a retry could duplicate an order). Every method raises `DhanApiError` on a
non-2xx response or malformed body - callers decide how to degrade.

This client makes real network calls. It is never imported by anything in
`quant_intelligence/tests/` in a way that hits the network - tests mock
`requests.post`/`requests.get` and assert the constructed request.
"""
from __future__ import annotations

import time

import requests

from quant_intelligence.brokers.dhan_rate_limit import LIMITER
from quant_intelligence.config.credentials import token_status
from quant_intelligence.config.settings import SETTINGS

# Expiry dates change at most daily, so cache them and halve the option-chain request rate.
_EXPIRY_CACHE_TTL_SEC = 30 * 60
_expiry_cache: dict[tuple, tuple[float, list[str]]] = {}


class DhanApiError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class DhanRateLimited(DhanApiError):
    """Dhan said 429 repeatedly; reads are paused briefly (no request was sent, or the last one
    pushed the circuit breaker open). `retry_in` is the seconds until requests resume."""

    def __init__(self, message: str, retry_in: float = 0.0):
        super().__init__(message, status_code=429)
        self.retry_in = retry_in


def _retry_after_seconds(resp) -> float | None:
    try:
        return float(resp.headers.get("Retry-After"))
    except (TypeError, ValueError, AttributeError):
        return None


def clear_expiry_cache() -> None:
    _expiry_cache.clear()


def _plain(text: str, limit: int = 160) -> str:
    """A gateway error page (CloudFront answers 502/504 with a full HTML document) as one short line."""
    import re

    if "<" not in text:
        return text
    stripped = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()
    return stripped[:limit] + ("..." if len(stripped) > limit else "")


class DhanApiClient:
    def __init__(self, client_id: str | None = None, access_token: str | None = None, base_url: str | None = None):
        self._follows_settings = client_id is None and access_token is None  # picks up renewed tokens by itself
        self.client_id = client_id or SETTINGS.dhan_client_id
        self.access_token = access_token or SETTINGS.dhan_access_token
        self.base_url = (base_url or SETTINGS.dhan_base_url).rstrip("/")

    def is_configured(self) -> bool:
        return bool(self.client_id and self.access_token)

    def _headers(self) -> dict:
        return {
            "access-token": self.access_token,
            "client-id": self.client_id,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _require_configured(self) -> None:
        if not self.is_configured():
            raise DhanApiError("DhanApiClient is not configured: set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in .env")

    def _throttle(self, category: str) -> None:
        if category == "option_chain":
            self._throttle_option_chain()
        else:
            LIMITER.acquire(category)

    def _request(self, method: str, path: str, body: dict | None = None, *, category: str, retry: bool = True,
                 timeout: float = 15) -> dict:
        """One Dhan call: breaker check -> spacing -> request -> 429 backoff/retry (reads only)."""
        if self._follows_settings:
            self.client_id, self.access_token = SETTINGS.dhan_client_id, SETTINGS.dhan_access_token
        self._require_configured()
        limits = SETTINGS.dhan
        attempts = limits.max_attempts if retry else 1
        status = token_status(self.access_token)
        if self._follows_settings and status["state"] == "expired":
            # One automatic recovery (renew/generate) before giving up; the keeper rate-limits its own attempts.
            from quant_intelligence.brokers.dhan_auth import TOKEN_KEEPER

            if TOKEN_KEEPER.ensure_fresh() in ("renewed", "generated"):
                self.client_id, self.access_token = SETTINGS.dhan_client_id, SETTINGS.dhan_access_token
                status = token_status(self.access_token)
        if status["state"] == "expired":
            raise DhanApiError(
                f"Your Dhan access token expired on {status['expires_at']:%d %b %H:%M} IST (tokens last about 24 hours). "
                "Generate a new one in your Dhan account and enter it under API Keys.",
                status_code=401,
            )
        for attempt in range(1, attempts + 1):
            blocked = LIMITER.auth_block_remaining(self.access_token)
            if blocked > 0:
                raise DhanApiError(
                    f"Dhan rejected your credentials (401); not sending requests for another {blocked:.0f}s. "
                    "Update the access token (tokens expire about every 24 hours).",
                    status_code=401,
                )
            remaining = LIMITER.cooldown_remaining(category)
            if remaining > 0:
                raise DhanRateLimited(
                    f"Dhan rate limit: {category} requests paused for another {remaining:.0f}s", retry_in=remaining
                )
            self._throttle(category)
            url = f"{self.base_url}{path}"
            if method == "GET":
                resp = requests.get(url, headers=self._headers(), timeout=timeout)
            else:
                resp = requests.post(url, headers=self._headers(), json=body, timeout=timeout)
            try:
                data = self._handle(resp)
            except DhanApiError as e:
                if e.status_code == 401:
                    LIMITER.record_auth_failure(self.access_token)
                if e.status_code == 429:
                    retry_after = _retry_after_seconds(resp)
                    opened = LIMITER.record_429(category, retry_after)
                    if opened:
                        raise DhanRateLimited(
                            f"Dhan rate limit hit on {category} calls; paused for {LIMITER.cooldown_remaining(category):.0f}s",
                            retry_in=LIMITER.cooldown_remaining(category),
                        ) from e
                    if retry and attempt < attempts:
                        time.sleep(max(retry_after or 0.0, limits.backoff_sec * attempt))
                        continue
                raise
            LIMITER.record_success(category)
            return data
        raise AssertionError("unreachable")

    def _get(self, path: str, category: str = "non_trading", retry: bool = True) -> dict:
        return self._request("GET", path, category=category, retry=retry)

    def _post(self, path: str, body: dict, category: str = "non_trading", retry: bool = True, timeout: float = 15) -> dict:
        return self._request("POST", path, body, category=category, retry=retry, timeout=timeout)

    def _handle(self, resp: "requests.Response") -> dict:
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if not resp.ok:
            message = data.get("remarks") or data.get("message") or _plain(resp.text) or f"HTTP {resp.status_code}"
            raise DhanApiError(f"Dhan API error ({resp.status_code}): {message}", status_code=resp.status_code)
        return data

    def _throttle_option_chain(self) -> None:
        LIMITER.acquire("option_chain")

    def _post_option_chain(self, path: str, body: dict) -> dict:
        """POST to an option-chain endpoint (shared spacing; 429 backoff and retry)."""
        return self._post(path, body, category="option_chain")

    def get_expiry_list(self, underlying_scrip: int, underlying_seg: str) -> list[str]:
        key = (self.base_url, int(underlying_scrip), underlying_seg)
        cached = _expiry_cache.get(key)
        if cached and time.monotonic() - cached[0] < _EXPIRY_CACHE_TTL_SEC:
            return list(cached[1])
        # UnderlyingScrip must be an integer: the scrip master yields string ids, which Dhan rejects (814).
        body = self._post_option_chain(
            "/optionchain/expirylist", {"UnderlyingScrip": int(underlying_scrip), "UnderlyingSeg": underlying_seg}
        )
        expiries = body.get("data", [])
        if expiries:  # never cache an empty answer (e.g. a future with no live options yet)
            _expiry_cache[key] = (time.monotonic(), list(expiries))
        return expiries

    def get_option_chain(self, underlying_scrip: int, underlying_seg: str, expiry: str) -> dict:
        return self._post_option_chain(
            "/optionchain",
            {"UnderlyingScrip": int(underlying_scrip), "UnderlyingSeg": underlying_seg, "Expiry": expiry},
        )

    def get_ltp(self, segment_map: dict[str, list[int]]) -> dict:
        return self._post("/marketfeed/ltp", segment_map, category="quote")

    def get_fund_limit(self) -> dict:
        return self._get("/fundlimit", category="non_trading")

    def get_positions(self) -> list[dict]:
        data = self._get("/positions", category="non_trading")
        return data if isinstance(data, list) else data.get("data", [])

    def place_order(self, payload: dict) -> dict:
        # Throttled but never auto-retried: a retried order could be placed twice.
        return self._post("/orders", payload, category="orders", retry=False)

    def get_historical_daily(
        self, security_id: str, exchange_segment: str, instrument: str, from_date: str, to_date: str
    ) -> dict:
        """Daily OHLCV candles. `from_date`/`to_date` as 'YYYY-MM-DD'."""
        return self._post(
            "/charts/historical",
            {
                "securityId": security_id,
                "exchangeSegment": exchange_segment,
                "instrument": instrument,
                "expiryCode": 0,
                "oi": False,
                "fromDate": from_date,
                "toDate": to_date,
            },
            category="data",
        )

    def get_intraday_minute(
        self,
        security_id: str,
        exchange_segment: str,
        instrument: str,
        interval: str,
        from_date: str,
        to_date: str,
    ) -> dict:
        """Intraday OHLCV candles. `interval` is one of '1'/'5'/'15'/'25'/'60' (minutes).
        `from_date`/`to_date` as 'YYYY-MM-DD HH:MM:SS'. Dhan caps this at 90 days per call."""
        return self._post(
            "/charts/intraday",
            {
                "securityId": security_id,
                "exchangeSegment": exchange_segment,
                "instrument": instrument,
                "interval": interval,
                "oi": False,
                "fromDate": from_date,
                "toDate": to_date,
            },
            category="data",
        )
