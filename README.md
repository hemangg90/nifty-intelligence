# Quant Strategy Intelligence & Execution System (`nifty-intelligence`)

A research-first trading terminal for Indian **index options (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, SENSEX)**, **stock options**
(a fixed 15-stock watchlist) and **MCX commodity options**, built as a Streamlit dashboard on top of
the Dhan broker API.

It answers one question, over and over, for every instrument it watches:

> **Which validated strategy has the strongest statistically validated edge under today's market
> conditions - and is there a valid setup right now? If not, do nothing.**

Everything is paper-traded by default. Real orders require a deliberate, double-gated opt-in (see
[Safety model](#safety-model)). **This is research software, not financial advice - nothing here is a
promise of profit** (read [Known limitations](#known-limitations-read-this-before-trusting-a-signal)).

---

## Table of contents

1. [What it does, in one picture](#what-it-does-in-one-picture)
2. [Core ideas](#core-ideas)
3. [How a trade happens, end to end](#how-a-trade-happens-end-to-end)
4. [Dashboard pages](#dashboard-pages)
5. [The strategy library](#the-strategy-library)
6. [Market data](#market-data)
7. [Research layer: features, regimes, backtests, analogues, ranking](#research-layer)
8. [Options layer](#options-layer)
9. [Risk engine and position sizing](#risk-engine-and-position-sizing)
10. [Execution: paper, auto-trading, live](#execution-paper-auto-trading-live)
11. [Safety model](#safety-model)
12. [Quick start](#quick-start)
13. [Configuration reference](#configuration-reference)
14. [Deploying on Streamlit Community Cloud](#deploying-on-streamlit-community-cloud)
15. [Project layout](#project-layout)
16. [Database](#database)
17. [Testing](#testing)
18. [Troubleshooting](#troubleshooting)
19. [Extending it](#extending-it)
20. [Known limitations](#known-limitations-read-this-before-trusting-a-signal)

---

## What it does, in one picture

```
 Dhan API / your CSVs  (real data only - no synthetic fallback)
          |
          v
   DataManager ---- quality gate (gaps, bad OHLC, out-of-session bars, staleness) ----> FAIL/DEGRADED = NO TRADE
          |
          v
   Feature engine (trend, ATR, VWAP, opening range, RSI, Bollinger, pivots ...) -- no look-ahead
          |
          +--> Regime engine (10 regimes, probabilities, not a single label)
          |
          v
   For each of the 17 strategies:  historical backtest  ->  "analogues" (30 most similar past setups)
                                   ->  conditional expected R, win probability, confidence
          |
          v
   Ranking engine ----> best strategy, or  NO TRADE  (first-class outcome)
          |
          v
   Setup detector: does the chosen strategy's entry trigger on the latest bar?   (WAITING / TRIGGERED)
          |
          v
   Option selector (live Dhan option chain)  ->  contract (CE/PE, strike, expiry)
          |
          v
   Premium model: underlying stop/target  ->  option-premium stop/target
          |
          v
   Position sizing (risk % and capital cap)  ->  RISK ENGINE (can VETO)  ->  Paper broker  (or Dhan, if LIVE-authorised)
          |
          v
   Monitor stop/target on live premium, square off before the close, record everything in the database
```

## Core ideas

- **Research first, execution second.** A strategy must *earn* the right to trade today by having
  historically worked in conditions similar to today's. Picking a strategy and *entering a trade* are
  separate steps: a selected strategy sits in `WAITING_FOR_SETUP` until its own entry conditions fire.
- **NO TRADE is a result, not a failure.** The system refuses to trade when data quality is not OK,
  no strategy clears a minimum edge, confidence is too low, or the top two strategies are
  indistinguishable.
- **Real data only.** There is no synthetic or random fallback. If Dhan (or your CSVs) can't supply
  candles, the system raises `DataUnavailableError` and says why, rather than inventing prices.
- **Everything is traded as options.** Indices, stocks and commodities all resolve to an option
  contract from the live chain. (Strategies are *researched* on the underlying's price action and
  *executed* in option premium - see limitations.)
- **A deterministic risk engine has the last word.** It does not depend on the research layer and
  cannot be overridden by it.
- **Transparent, not a black box.** Scores expose their components, analogues are inspectable, regimes
  are rule-based with visible probabilities, and every order, fill, position and risk decision is
  stored with an ID.

## How a trade happens, end to end

1. **Scan.** For each instrument the pipeline loads ~60 days (configurable, 10-180) of 5-minute
   candles, validates them, computes features and classifies the regime.
2. **Rank.** Every strategy is back-tested over that history. The 30 historical setups most similar to
   *now* give each strategy a conditional expected R, win probability and confidence. The ranking
   engine picks a winner or says NO TRADE.
3. **Detect.** The setup detector checks whether the winner's entry condition is true on the latest
   bar (some strategies also need the live option chain).
4. **Resolve a contract.** The live Dhan option chain is fetched; the option selector maps
   LONG -> call, SHORT -> put (BUY) or the mirrored SELL leg, picks the strike (ATM by default,
   configurable moneyness offset) and expiry (nearest by default).
5. **Translate to premium.** The underlying's stop/target distance is scaled by the option's live
   delta into a premium stop/target. A SELL *requires* a valid premium stop (unbounded risk otherwise).
6. **Size.** Quantity (whole lots) is the smaller of a risk-based size (90% of the max risk per trade)
   and a capital cap (max % of equity in one trade). A BUY that costs more than the available cash is
   refused ("Insufficient funds").
7. **Approve.** The risk engine runs its checks in order and either approves or vetoes (logged with a
   decision ID and reason).
8. **Execute.** Paper broker simulates a fill with adverse slippage (or Dhan places a real order, only
   if LIVE is fully authorised).
9. **Manage.** Each cycle, every open option's live premium is compared with its stop/target; positions
   are squared off shortly before the session closes.

## Dashboard pages

Run the app and use the sidebar page list. The main page (`app.py`) is the command center.

| Page | Purpose |
|---|---|
| **Main (`app.py`)** | Instrument/timeframe/lookback controls, Dhan API-key panel, trading-mode badge, the current decision, strategy ranking table and regime probabilities. |
| 01 Market State | The latest feature vector (every feature is documented) and data-quality status. |
| 02 Strategy Intelligence | Global vs conditional (current-state) performance for every strategy. |
| 03 Strategy Library | Each strategy's description, parameters and required features. |
| 04 Backtest Lab | Event-driven backtests with in-sample / out-of-sample split and walk-forward folds; editable parameters. |
| 05 Regime Analysis | Regime probabilities and classifier confidence over time. |
| 06 Historical Analogues | The exact past setups used to score each strategy today. |
| **07 Paper Trading** | A trading terminal for one instrument: quote header (last price, day change, range), candlestick chart with VWAP/EMA overlays and the setup's entry/stop/target lines, **option-chain ladder** (calls | strike | puts, OI, change in OI, IV, ATM highlighted) with an open-interest chart, PCR, max pain and support/resistance, the **strategy leaderboard**, a decision card (regime, selected strategy, setup status) and an **order ticket** showing premium, quantity, amount needed, max loss/profit **after charges**, risk:reward and a funds meter before you submit. "All instruments" scope hands over to the background scanner. |
| **08 Positions & Orders** | A broker-style view of the paper account. Live KPIs (day P&L net of charges, capital used/available, win rate); tabs for **open positions** (cards with live P&L, stop-to-target progress bar, charges and a manual **Exit** / **Exit all** with confirmation), today's trades, filterable trade history, order book, fills (with slippage cost) and **analytics** (equity curve, profit factor, expectancy, payoff, drawdown, streaks, charges breakdown, P&L by strategy/instrument). CSV downloads on every table. |
| 09 Risk Control | Account state, configured limits, the **emergency kill switch**, recent risk events. |
| 10 Research Reports | Structured reports generated only from stored system data. |
| 11 System Logs & Health | Dhan connectivity/credentials status and system logs. |
| **12 Live Options Trading** | Real orders via Dhan - disabled unless fully authorised (see Safety model). |
| **13 Auto Multi-Instrument Trading** | Background auto-trader over NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, SENSEX and the 15-stock watchlist: control bar, live account/market tiles, a **market watch** (last price, change %, day-range bar, sparkline and the last scan result per instrument), and tabs for positions, scan results, an activity feed (orders, risk vetoes, exits) and today's performance. |
| **15 Daily Report** | How the system worked each day and which strategies are earning, generated automatically after each market close and kept day by day: a plain-English summary, a strategy leaderboard with confidence (mean R with 95% whiskers - hollow marker means too few trades), signals-vs-trades funnel and why the system stood aside, realised vs backtest-expected R, breakdowns (instrument, exit reason, entry hour, TIE-BREAK vs normal ...), system/data health, and a table of every past day. Downloads: Excel, Markdown, JSON. |
| **16 NSE Heatmap** | Every index, stock and commodity on the watchlist as a treemap coloured by % change (today, 5 days or 1 month), with gainer/loser tables and a **Selection panel** that picks the day's biggest movers for the auto paper traders (see *Heatmap mover selection*). |
| **14 Commodity Auto Trading (MCX)** | The same layout for a second, independent background auto-trader for MCX commodities with its own session hours and square-off countdown. |

A sidebar indicator on every page shows when auto-trading is running and offers a Stop button.

**Look and feel.** Every trading page starts with a status bar (IST clock, NSE and MCX session open/closed with a
countdown, Dhan connection state, whether each auto-trader is running, PAPER/LIVE mode, kill switch). Numbers use
Indian grouping (lakh/crore) and rupee signs; P&L is always colour **and** triangle **and** sign, never colour alone.
Charts follow a validated colour palette (blue/orange/aqua for series; green/red reserved for good/bad), one axis per
chart, hairline grids and trading-hour gaps removed from intraday candles. The theme lives in `.streamlit/config.toml`
and `ui/theme.py`. Quotes in watchlists come from the on-disk candle cache, so they cost no Dhan requests.
Charges shown are estimates at Dhan's brokerage and the statutory option rates (display only - the paper broker books
gross premium P&L).

## The strategy library

17 strategies implement one interface (`strategies/base_strategy.py`): `generate_historical_setups`,
`check_setup` (live), and a bar-by-bar `simulate_trade` (stop checked before target within a bar -
conservative). Default exits are ATR-based.

| Strategy | Idea |
|---|---|
| Opening Range Breakout | Break of the first-30-minute range. |
| Momentum | Strong recent momentum confirmed by trend and volume. |
| VWAP Mean Reversion | Fade stretched moves away from session VWAP in weak trends. |
| RSI(2) Mean Reversion | Connors-style pullbacks in the direction of the EMA(200) trend. |
| Bollinger Band Mean Reversion | Fade closes outside the bands that reclaim them, in ranges. |
| Supertrend Trend Following | Supertrend(10, 3) direction flips, trailing the line. |
| EMA 50/200 Crossover | Golden/death cross trend following. |
| Donchian Channel Breakout | 20-bar breakout with cooldown. |
| MACD Signal Crossover | MACD/signal cross filtered by EMA(200) direction. |
| CPR Breakout | Break of the prior session's Central Pivot Range. |
| Camarilla Pivot Reversal | Fade prior-session R3/S3 touches toward the pivot. |
| Inside Bar Breakout | Break of an inside bar after a coil. |
| Gap Fill Reversion | Fade a moderate opening gap back toward the prior close. |
| PCR Extreme Contrarian * | Fade chain-wide put/call-ratio extremes with an RSI(2) reversal. |
| OI Buildup Confirmation * | Donchian breakout confirmed by rising near-ATM open interest. |
| IV Crush Premium Writer * | Sell an OTM option when ATM IV is rich vs realised vol. |
| Unusual Volume Fade * | Fade a single-strike volume spike near the money. |

`*` **Chain-aware** strategies need a live option chain. They return no historical setups (there is no
historical chain data), so they can fire live but are never selected by the ranker. See limitations.

## Market data

Preference order, per instrument: **CSV files you provide** (`data_cache/csv/<SYMBOL>_<TIMEFRAME>.csv`
with `timestamp,open,high,low,close,volume`) -> **Dhan** -> error. Results are cached as Parquet (and refreshed only when a new bar closes - see *Staying under Dhan's rate limits*) in
`data_cache/parquet_cache/`.

- **Instruments:** NSE equities, NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, SENSEX (Dhan `INDEX` candles; SENSEX options route to `BSE_FNO`) and MCX futures underlyings.
- **Intraday intervals** from Dhan: 1, 5, 15, 25, 60 minutes, up to ~90 days per request; otherwise daily.
- **Timestamps:** Dhan returns UTC epoch seconds. The adapter proves the timezone from the data
  itself (bars must fall inside the trading session), shifts to naive IST, and refuses data that fits
  no timezone. All "now" values use an IST clock (`utils/timeutil.py`) because Streamlit Cloud runs in UTC.
- **Forming candle:** the last, still-open bar is dropped so signals never repaint.
- **Stray bars:** a few pre/post-close prints (e.g. zero-volume 15:30 bars) are dropped; if more than 2%
  of bars are out of session the data is **FAIL**.
- **Quality gate** (`data/quality.py`): duplicates, intra-session gaps, zero/negative prices, impossible
  OHLC, invalid volume, staleness against the real market calendar. `OK` is required to trade;
  `DEGRADED`/`FAIL` force NO TRADE and the risk engine vetoes anyway.
- **No volume on indices.** Dhan gives index candles no volume, so volume-dependent strategies
  (Momentum, VWAP Mean Reversion, Unusual Volume Fade) are skipped for NIFTY/BANKNIFTY.
- **Sessions and holidays:** NSE 09:15-15:30 IST, MCX 09:00-23:55 IST (`utils/market_profile.py`). Weekends and
  exchange holidays are skipped when judging staleness, market-open state and runner gating. Built in: the
  fixed-date holidays (26 Jan, 15 Aug, 2 Oct, 25 Dec for NSE and MCX; 1 May for NSE only). Holidays that move
  every year (Holi, Diwali, Eid, Good Friday ...) are **not guessed**: add them to
  `quant_intelligence/config/market_holidays.txt` (`YYYY-MM-DD [NSE|MCX] name`, picked up without a restart).
  A day missing from the list is treated as a trading day, so keep it current from the exchanges' circulars.

### Keeping data fresh by itself (data keeper)

Candles used to be downloaded only when a runner scanned or a page ran. With the auto-traders stopped and
nobody on the page they aged past the 30-minute tolerance and every decision became "data quality DEGRADED -
NO TRADE" until someone pressed a button. A background **data keeper** (`data/data_keeper.py`, starts with the
first page load or runner start) now refreshes all 25 watchlist instruments every `DATA_KEEPER_INTERVAL_SEC`
(60 s). It is a no-op when the cache already holds the latest closed bar; otherwise it costs one small tail
request, spaced by the Dhan rate limiter. Status is in the top status bar ("Data feed: current / N stale"), a
banner on the NSE/MCX panels naming the reason, and a table with a **Refresh candles now** button on the
*System Logs & Health* page. Pages also recompute their cached analysis when the credentials change, when the
last result was not OK for a minute, or when a new bar has closed - saving a new token clears stale results
immediately. Candle files are written atomically and per instrument one fetch runs at a time, so the keeper and the pages never corrupt or duplicate each other's downloads. Disable with `DATA_KEEPER=false`.

### Token lifecycle (token keeper)

Dhan access tokens last 24 hours. The **token keeper** (`brokers/dhan_auth.py`) checks the token's own expiry
(no network) and:

1. **Renews** it with Dhan's `RenewToken` when under `TOKEN_RENEW_BEFORE_HOURS` (8) remain - needs no extra setup,
   but only works while the token is still active.
2. If `DHAN_PIN` and `DHAN_TOTP_SECRET` are set, **generates** a new token from a time-based one-time code (RFC 6238,
   standard library) - this also recovers from a token that has already expired.

It makes at most one attempt at a time, waits 10 minutes after a failure (a wrong PIN must not lock the account),
never puts the PIN/TOTP in logs or error messages, and applies the new token to the running app (and `.env` when
writable). The status shows in *API Keys*. **Treat `DHAN_PIN` and `DHAN_TOTP_SECRET` like passwords:** put them in
Streamlit Secrets or `.env`, never in git or chat; they let anyone who has them mint tokens for your account. On
Streamlit Cloud the sidebar only changes the running app (lost on reboot) - put lasting values in the app's Secrets.
Disable with `TOKEN_KEEPER=false`. The renew/generate calls follow Dhan's documented endpoints but could only be
tested against mocked responses, not a live account - confirm the first renewal in the log (`token_keeper`).
- `data_adapters/synthetic.py` still exists but is **used only by the test-suite**, never at runtime.

## Research layer

- **Features** (`features/feature_engine.py`): returns, trend slope, momentum, market structure, ATR and
  ATR percentile, realised vol and vol expansion, relative volume, session-reset VWAP and distance,
  opening range, gap, time since open, RSI(2/14), Bollinger bands, EMAs, MACD, Supertrend, Donchian,
  CPR and Camarilla pivots. Every feature uses only data up to its own bar (backward-looking by
  construction). Derivatives fields (OI, PCR, IV, futures basis) are NaN placeholders by design.
- **Regimes** (`regimes/regime_engine.py`): rule-based scores turned into a softmax probability
  distribution over TREND_UP, TREND_DOWN, RANGE, COMPRESSION, VOL_EXPANSION, VOL_CONTRACTION,
  HIGH_VOL, LOW_VOL, EVENT_DRIVEN, LIQUIDITY_STRESS. No machine learning.
- **Backtesting** (`backtesting/engine.py`): sequential, non-overlapping trades per strategy, 75-bar
  maximum hold. Costs follow how the trade is actually made - as an **option**: Dhan's Rs 20 per order
  (x2) plus STT (sell-side premium), exchange charge, SEBI fee, stamp duty and 18% GST on *premium*
  turnover, plus slippage; NSE and MCX use their own STT/exchange rates. Because the backtest only has
  underlying prices, the premium is an assumption (`OPTION_PREMIUM_PCT` of the underlying, `OPTION_DELTA`).
  Ranking works in **net-of-cost R**. Supports in-sample / out-of-sample splits and rolling walk-forward.
- **Analogues** (`analogues/analogue_engine.py`): a transparent k-NN. The current market state is
  compared (standardised distance) against the entry-time features of every past setup using six
  features - `trend_slope`, `momentum_20`, `atr_percentile_100`, `vol_expansion`, `relative_volume`,
  `vwap_distance_pct`. The 30 nearest give conditional expected R, win rate, median R and confidence.
- **Ranking** (`ranking/ranking_engine.py`, shared with the live pipeline through
  `research/ranking_core.py`): a visible score
  `0.5*edge + 0.3*(P(win)-0.5) + 0.3*confidence_weight - 0.2*drawdown_penalty - cost_penalty`, every
  component exposed. The `edge` is a **shrunk lower confidence bound** of the conditional net-of-cost
  expected R (shrunk toward the strategy's own average, then minus one standard error). Confidence is
  based on how many *independent trading days* the analogues span, and the standard error is
  cluster-robust by day. Drawdown is measured in R. A strategy is *eligible* only with a shrunk edge
  >= 0.05, confidence >= MEDIUM, and (with enough history) a positive edge in at least half of four
  chronological folds. NO TRADE is returned if data quality is not OK, nothing is eligible, or the top two
  are not statistically distinguishable (edge difference < 1 combined standard error).
- **Ranker evaluator** (`ranking/evaluate_ranker.py`): a walk-forward test of the ranker itself. At each
  decision point it ranks using only trades already closed, then compares the selected strategy's realised
  net R with picking a strategy at random. Run `python -m quant_intelligence.ranking.evaluate_ranker`
  (uses the candles already in `data_cache/`; no network).

## Options layer

- **Chain analytics** (`options/chain_analytics.py`): parses Dhan's chain into strikes with LTP, IV,
  OI, greeks, ATM strike, PCR and ATM IV.
- **Contract selection** (`options/option_selector.py`): indices are registered statically
  (NIFTY id 13, BANKNIFTY id 25, segment `IDX_I`); NSE F&O stocks and MCX underlyings resolve
  dynamically from Dhan's scrip master (lot size and strike step refreshed from it). MCX options sit on
  the front-month future, falling back to the next month when the front has no live expiry.
- **Premium model** (`options/premium_model.py`): scales underlying stop/target by live |delta|
  (0.5 if unknown) - a first-order approximation that ignores gamma/theta, so it is an estimate to be
  monitored against live premium, not a guarantee.
- **Staying under Dhan's rate limits.** Dhan's limits apply to the whole *account*, and exceeding them
  (HTTP 429, code 805) risks the account being blocked. Published limits: orders 10/s, data (candles) 5/s,
  quote (live prices) **1/s**, non-trading (funds, positions) 20/s, option chain 1 per 3 s. The app is built
  to stay inside them rather than recover after hitting them:
  - **One limiter for everything** (`brokers/dhan_rate_limit.py`): every Dhan call is classed into one of
    those categories and spaced ~30% under the limit, shared by both background runners and every browser tab.
  - **Chains only when needed:** the option chain is fetched only for an instrument whose strategy has a
    *triggered setup* (or a chain-aware strategy), not for all ~25 instruments every cycle.
  - **Shared single-flight caches** (`brokers/dhan_cache.py`): live prices (2 s) and option chains (15 s) are
    fetched once and shared, so ten open tabs cost the same as one. Expiry lists are cached for 30 minutes.
  - **Candles refresh on bar close:** an instrument is re-requested only when a new bar has closed (about one
    small tail request per instrument per bar), then merged into the cache.
  - **429 handling:** read calls honour `Retry-After` or back off (6/12 s) and retry up to 3 times. After
    repeated 429s a short circuit breaker (30 s, doubling to 2 min) fast-fails reads without touching the
    network and then resumes by itself. **Orders are throttled but never auto-retried** (a retry could
    duplicate an order) and are never blocked by the breaker, so exits can always go out.
  - **Rejected token (401):** nothing more is sent with that token for 5 minutes or until you paste a new
    one; hammering Dhan with a dead token on every cycle would itself risk a block.
  - **Visibility:** *System Logs & Health* shows calls per category over the last minute, 429 counts and any
    pause; a sidebar banner appears on every page while requests are paused.

  The app cannot see requests made by *other* processes on the same account (a second copy of the app, a
  local run plus a cloud deployment, scripts). **Run one instance at a time.**

## Risk engine and position sizing

### How loss per trade is set

The rupee loss of a stopped trade is fixed by **position size, not by how wide the stop is**: the quantity is chosen so
that (stop distance x quantity) fits `MAX_RISK_PER_TRADE_PCT` of equity, rounded down to whole lots. A wider stop simply
buys fewer lots. At the default 0.2% that is about **Rs 1,800 on Rs 10 lakh** (it scales with equity and is halved
after two losers in a row or once half the daily limit is gone). Two consequences to know:

* **Whole lots.** One NIFTY lot (75) with a 15-point premium stop risks about Rs 1,100, so a stopped NIFTY trade loses
  about Rs 1,100, not exactly Rs 1,800. The budget is a ceiling.
* **Big lots can be unaffordable.** If even ONE lot's stop-loss exceeds the budget (many stock options, NATURALGAS,
  COPPER ...) the setup is skipped, never forced to one lot. The decision log files it as `NO_SIZE`, and the Daily Report
  names those instruments. A tighter stop (`STOP_ATR_SCALE` below 1) or a bigger account lets them fit.

Environment variables override these defaults: if you set `MAX_RISK_PER_TRADE_PCT` / `MAX_DAILY_LOSS_PCT` in `.env` or in
Streamlit **Secrets**, change them there too.

`risk/risk_engine.py` is deterministic, independent of the research layer, and records every
approval/veto (with reason and per-check results) to the `risk_events` table. Checks run in this order:

1. Emergency kill switch  2. Broker connected  3. Data quality OK  4. Daily loss limit
5. Max drawdown from peak  6. Profit giveback lock  7. Losing-streak cooldown  8. Max trades per day
9. Max open positions  10. Strategy daily loss stop  11. Max risk per trade (x size multiplier)
12. Max exposure per strategy  13. Max total portfolio exposure  14. Correlated-group positions and risk
15. Liquidity (relative volume >= 0.3)

Defaults (all overridable by environment variable): risk per trade **1%**, daily loss **3%**, drawdown
**8%**, trades per day **6**, per-strategy exposure **10%**, portfolio exposure **20%**, capital in one
trade **5%** of equity, **4** open positions, **2** same-direction positions and **2%** open risk per
correlated group, a **60-minute** pause after **3** losers in a row, a strategy stops for the day after
**2** losses, and no new trades once a day that reached **+1.5%** gives back **half** of it.

**Equity and drawdown.** Equity is **marked to market** (cash + unrealised P&L from the position
monitor's live prices), so open losses count toward the daily-loss and drawdown limits before they
close. The drawdown peak is carried **across days and restarts** (rebuilt from all closed trades);
reset it on purpose from the Risk Control page.

**Correlation groups** (`config/risk_groups.py`): all index underlyings are one group; watchlist stocks
are grouped by sector (the four banks are one group); MCX commodities by sector.

**Size multiplier** (`risk_engine.risk_multiplier`): the per-trade risk budget is scaled by the smallest
of: a drawdown throttle (100% below 3% drawdown, falling linearly to 25% at the 8% hard stop), 50% once
half the daily loss limit is used, and 50% after 2 losing trades in a row. The Risk Control page shows
the current multiplier and why.

**Sizing** (`execution/auto_trader.py::size_position`): `quantity = floor(0.9 x risk budget x size
multiplier / stop distance)` in whole lots, then capped so premium x quantity <= the per-trade capital cap.

**Capital accounting** (`execution/capital.py`): the paper broker only moves cash when a position
*closes*, so capital used is derived from open BUY positions (entry premium x quantity) and
`available = cash - capital used`. Sold-option margin is **not** modelled; SELL premium is reported
separately.

## Execution: paper, auto-trading, live

- **Paper broker** (`brokers/paper_broker.py`): simulated fills with adverse slippage
  (`SLIPPAGE_TICKS x TICK_SIZE`), orders/fills/positions persisted.
- **Background engine** (`execution/engine.py`): a process-wide `TradingEngine` owns the shared broker,
  daily risk counters, the kill switch and the NSE scan loop; a second `ScanRunner` handles MCX
  commodities. Both run in daemon threads, **independent of any page or browser tab** - switching pages
  or closing the tab does not stop them; only **Stop**, a server restart or the host sleeping does. They
  share one cash pool, one risk-counter set and one kill switch.
- **Per cycle** (default every 60 s): scan every instrument -> pipeline -> chain -> setup -> size -> risk ->
  paper order; then check every open option's live premium against stop/target. A failed cycle is logged
  and retried; it never kills the worker.
- **Session handling:** by default trades only inside each market's session; positions are **squared off
  ~5 minutes before the close** at the latest price (also on the first cycle after a restart, if missed).
- **Restart recovery:** on start-up, today's positions, realised P&L and trade counts are rebuilt from the
  database; open rows from earlier days are marked `STALE`, never deleted.
- **Live trading** (`brokers/dhan_broker.py`, page 12): real orders through Dhan's v2 Orders API, only
  when fully authorised (below). It has been tested against Dhan's documented request shapes but **not
  against a real funded account**.

### Heatmap mover selection

The **NSE Heatmap** page (16) turns the market into a treemap (group, then sector, coloured by % change; blue = up,
orange = down so green/red keep meaning profit/loss; every tile also carries an arrow and a signed %). Its Selection
panel lets the two auto paper traders scan only the day's movers:

- **Full NSE market.** For NSE stocks the movers come from NSE's own live data - the top 20 gainers and top 20 losers
  among **every stock that has listed options** (about 200), via nseindia.com's `live-analysis-variations` endpoint
  (`data_adapters/nse_heatmap.fetch_fno_movers`, refreshed every `MOVER_NSE_REFRESH_MINUTES`, quote time taken from
  NSE). A selected stock does not have to be on the 15-stock watchlist: the pipeline fetches its candles and resolves its
  option chain from Dhan's scrip master like any other F&O underlying. Choose "Watchlist only" to restrict it.
  Indices and MCX commodities (NSE has no MCX data) still come from the candle cache.
- Per group (NSE indices, NSE stocks, MCX commodities) it takes the **N biggest gainers and N biggest losers**
  (default N = 5, at most 15 because NSE publishes 20 per side) by % change against the previous close, ignoring moves
  below `MOVER_MIN_ABS_PCT` (0.3%) - a quiet day selects fewer, or none.
- **Direction follows the move.** An up-mover may only take bullish setups (a bought call), a down-mover only bearish
  ones (a bought put). A setup against the move is skipped *before* the option chain is fetched, logged as
  `DIRECTION_FILTER`, and counted in the Daily Report. Written (SELL) options are not filtered.
- It only narrows what is scanned and which direction is allowed. The strategies still decide whether to enter, and the
  risk engine, loss limits and sizing are unchanged. Positions already open are still monitored and exited.
- "Keep refreshing" recomputes the list every scan cycle; otherwise it is fixed when you press Apply. **Clear** goes
  back to scanning everything. Pages 13/14 show a banner and a Heatmap column while it is on.
- **Honest fallbacks.** NSE's endpoints are undocumented and NSE blocks some servers (Streamlit Cloud included, possibly).
  If NSE is unreachable, or NSE is closed, the selection says so and uses the watchlist from the candle cache. If the
  newest quote/candle is not from today and within `MOVER_MAX_AGE_MINUTES` (20), **nothing is selected** - the system
  never trades on yesterday's move or on a missing quote treated as 0%. The map hides tiles whose data is not current
  while others are live. (The older `/api/equity-stockIndices` route used by `fetch_heatmap_universe` now returns 404
  from NSE for every index, so it is no longer used.)
- The map can show only the watchlist plus NSE's top 20 + 20 movers, not all ~200 stocks, because that is all NSE
  publishes live; the 5-day and 1-month views use the watchlist candles only.
- The state lives in the running process (not saved): after a restart, apply it again, or set
  `MOVER_SELECTION_ENABLED=true` so both traders start with it on.

This filter is **unmeasured**: there is no evidence yet that following the day's move improves net R. Compare a few
sessions in the Daily Report (TIE-BREAK/entry breakdowns and the filtered-setup count) before relying on it.

## Safety model

- **Paper by default.** `TRADING_MODE=PAPER`.
- **Live needs two independent switches**: `TRADING_MODE=LIVE` **and**
  `TRADING_LIVE_CONFIRM=YES_I_UNDERSTAND_THE_RISK`. `DhanBroker` re-checks this itself, so a bug elsewhere
  cannot enable real orders.
- **Kill switch** (page 09) blocks all new trades instantly.
- **Mandatory stop on every SELL;** BUY loss is capped at premium paid.
- **Risk engine veto** on every order; nothing downstream can override it.
- **Credentials:** the sidebar "API Keys" panel writes your token to a local `.env` (gitignored). On a
  shared/public deployment that panel would let any visitor change the token - keep such apps private
  and prefer platform Secrets.
- **Never commit `.env`.** It is in `.gitignore`.

## Quick start

Requires **Python 3.11+** (developed on 3.12) and a Dhan account with API access (Dhan Client ID and
access token). Check Dhan's current terms for what its market-data APIs require on your account.

```bash
git clone https://github.com/hemanggg90/nifty-intelligence.git
cd nifty-intelligence
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Create .env in the project root (see Configuration reference). Minimum:
#   DHAN_CLIENT_ID=your-client-id
#   DHAN_ACCESS_TOKEN=your-access-token

streamlit run quant_intelligence/app.py                # run from the project root
```

Open http://localhost:8501. You can also paste the Client ID/token into the sidebar's **API Keys**
panel instead of editing `.env`.

**First run walkthrough**

1. Open the dashboard; with valid credentials the pipeline runs on NIFTY and shows the ranking.
2. Go to **07 Paper Trading -> All instruments together -> Run one cycle now** to scan everything once.
3. Happy with the results? **Start auto trading**; watch **08 Positions & Orders** and **09 Risk Control**.

> Dhan **access tokens expire roughly every 24 hours**. Regenerate one in your Dhan account when you see
> `DH-901 / Invalid_Authentication`.

Helper scripts: `scripts/init_db.py` (create tables), `scripts/smoke_test.py` (run the pipeline once),
`scripts/ui_smoke_test.py` (headlessly execute every page and report exceptions).

## Configuration reference

Everything is an environment variable (loaded from `.env`); safe defaults let it run in PAPER mode with
SQLite even with no `.env`.

| Variable | Default | Meaning |
|---|---|---|
| `TRADING_MODE` | `PAPER` | `PAPER` or `LIVE`. |
| `TRADING_LIVE_CONFIRM` | *(empty)* | Must equal `YES_I_UNDERSTAND_THE_RISK` for live orders. |
| `DATABASE_URL` | local SQLite in `data_cache/` | Any SQLAlchemy URL (e.g. Postgres). |
| `DHAN_CLIENT_ID`, `DHAN_ACCESS_TOKEN` | *(empty)* | Dhan credentials. |
| `DHAN_BASE_URL` | `https://api.dhan.co/v2` | Dhan API base URL. |
| `OPTION_UNDERLYINGS` | `NIFTY,BANKNIFTY,FINNIFTY,MIDCPNIFTY,SENSEX` | Index underlyings scanned. |
| `OPTION_MONEYNESS_OFFSET` | `0` | 0 = ATM; +N = N strikes further OTM. |
| `OPTION_EXPIRY_PREFERENCE` | `NEAREST` | `NEAREST` or `NEAREST_MONTHLY`. |
| `OPTION_PRODUCT_TYPE` | `INTRADAY` | `INTRADAY`, `MARGIN` or `CNC` (order product type). |
| `DEFAULT_INSTRUMENT`, `DEFAULT_TIMEFRAME` | `NIFTY`, `5min` | Dashboard defaults. |
| `PAPER_STARTING_CAPITAL` | `1000000` | Paper account size (INR). |
| `AUTO_TRADE_REFRESH_SECONDS` | `60` | Pause between auto-trading cycles. |
| `LTP_REFRESH_SECONDS` | `5` | Live price refresh on the dashboard. |
| `EOD_SQUARE_OFF` | `true` | Close positions near each market's close. |
| `EOD_SQUARE_OFF_MINUTES` | `5` | How many minutes before the close. |
| `MAX_RISK_PER_TRADE_PCT` | `0.2` | What one stopped trade may cost, % of equity (Rs 2,000 on Rs 10 lakh; sizing uses 90% of it, about Rs 1,800). Quantity is whole lots, so the real loss is the budget rounded down to a lot. An instrument whose single lot already risks more does not trade (the report says so). See *How loss per trade is set*. |
| `MAX_CAPITAL_PER_TRADE_PCT` | `5.0` | Max premium outlay in one trade, % of equity (0 disables). |
| `MAX_DAILY_LOSS_PCT` | `1.5` | Daily loss limit; trading stops for the day beyond it. |
| `STOP_ATR_SCALE` | `1.0` | Multiplies the default stop and target distance of every ATR-based strategy together (0.5 = half as far). Explicit parameters are never scaled; clamped 0.2-3.0. |
| `MOVER_SELECTION_ENABLED` | `false` | Start both auto-traders scanning only the heatmap's movers (up-movers take calls, down-movers puts). Also switchable on the NSE Heatmap page. |
| `MOVER_TOP_N` | `5` | Gainers and losers selected per group (1-15). |
| `MOVER_MIN_ABS_PCT` | `0.3` | A move smaller than this is noise and is not selected. |
| `MOVER_MAX_AGE_MINUTES` | `20` | The newest candle/quote must be this recent (and from today) or nothing is selected. |
| `MOVER_UNIVERSE` | `nse` | `nse` = NSE stocks are chosen from every F&O stock on NSE's live gainers/losers (watchlist fallback if NSE is unreachable); `watchlist` = the 15 stocks only. |
| `MOVER_NSE_REFRESH_MINUTES` | `2` | How often NSE is asked for movers. |
| `MAX_STRATEGY_EXPOSURE_PCT` | `10.0` | Risk concentrated in one strategy. |
| `MAX_PORTFOLIO_EXPOSURE_PCT` | `20.0` | Total capital at risk. |
| `MAX_TRADES_PER_DAY` | `6` | Daily trade count limit. |
| `MAX_DRAWDOWN_PCT` | `8.0` | Max drawdown from peak equity (carried across days). |
| `DD_THROTTLE_START_PCT` | `3.0` | Drawdown at which position size starts shrinking. |
| `DD_THROTTLE_FLOOR` | `0.25` | Size multiplier reached at `MAX_DRAWDOWN_PCT`. |
| `LOSS_STREAK_THROTTLE_AFTER` | `2` | Half size after this many losers in a row today (0 disables). |
| `MAX_OPEN_POSITIONS` | `4` | Max positions open at once (0 disables). |
| `MAX_GROUP_POSITIONS` | `2` | Max same-direction positions per correlation group (0 disables). |
| `MAX_GROUP_RISK_PCT` | `2.0` | Max open risk per correlation group, % of equity (0 disables). |
| `LOSS_STREAK_PAUSE` | `3` | Losers in a row that pause new entries (0 disables). |
| `LOSS_STREAK_PAUSE_MIN` | `60` | Length of that pause, minutes. |
| `STRATEGY_MAX_LOSSES_PER_DAY` | `2` | Losses after which a strategy stops for the day (0 disables). |
| `PROFIT_LOCK_TRIGGER_PCT` | `1.5` | Day P&L (% of equity) that arms the profit lock (0 disables). |
| `PROFIT_LOCK_GIVEBACK` | `0.5` | Fraction of the day's peak P&L that, once given back, stops new trades. |
| `BROKERAGE_PER_ORDER_INR` | `20` | Dhan's flat brokerage per executed F&O order. |
| `SLIPPAGE_TICKS`, `TICK_SIZE` | `1`, `0.05` | Slippage in premium units; also used for paper fills. |
| `OPT_STT_SELL_RATE_NSE`, `OPT_STT_SELL_RATE_MCX` | `0.0015`, `0.0005` | STT on sell-side option premium. |
| `OPT_TXN_RATE_NSE`, `OPT_TXN_RATE_MCX` | `0.0003553`, `0.000418` | Exchange charge on premium turnover. |
| `SEBI_RATE`, `STAMP_BUY_RATE`, `GST_RATE` | `0.000001`, `0.00003`, `0.18` | SEBI fee, buy-side stamp duty, GST on brokerage + exchange + SEBI. |
| `DHAN_MIN_INTERVAL_QUOTE_SEC`, `DHAN_MIN_INTERVAL_DATA_SEC`, `DHAN_MIN_INTERVAL_NONTRADING_SEC`, `DHAN_MIN_INTERVAL_ORDERS_SEC`, `DHAN_MIN_INTERVAL_CHAIN_SEC` | `1.25`, `0.35`, `0.1`, `0.17`, `3.5` | Minimum seconds between Dhan calls per category (set under Dhan's limits of 1/s, 5/s, 20/s, 10/s, 1 per 3 s). Lower them only if you are sure of your account's limits. |
| `DHAN_429_MAX_ATTEMPTS`, `DHAN_429_BACKOFF_SEC` | `3`, `6` | Tries and base backoff for reads that get HTTP 429. |
| `DHAN_BREAKER_THRESHOLD`, `DHAN_BREAKER_COOLDOWN_SEC`, `DHAN_BREAKER_MAX_COOLDOWN_SEC` | `3`, `30`, `120` | Consecutive 429s that pause reads, and the pause length (doubles up to the max). |
| `DHAN_AUTH_FAILURE_COOLDOWN_SEC` | `300` | How long to stop sending requests with a token Dhan rejected (401). |
| `DHAN_QUOTE_CACHE_TTL_SEC`, `DHAN_CHAIN_CACHE_TTL_SEC` | `2`, `15` | How long a live-price / option-chain response is shared between tabs and the auto-traders. |
| `TIE_BREAK_PAPER`, `TIE_BREAK_LIVE`, `TIE_BREAK_SIZE_FACTOR` | `true`, `true`, `0.5` | Trade the top strategy (tagged TIE-BREAK, sized x factor) when the best two are statistically tied. |
| `DATA_KEEPER`, `DATA_KEEPER_INTERVAL_SEC` | `true`, `60` | Background candle refresh for every watchlist instrument. |
| `TOKEN_KEEPER`, `TOKEN_RENEW_BEFORE_HOURS` | `true`, `8` | Auto-renew the Dhan token when this many hours remain. |
| `DHAN_PIN`, `DHAN_TOTP_SECRET` | *(unset)* | Optional. With both set the token keeper can generate a new token even after expiry. **Secrets: never commit or paste them.** |
| `DATABASE_URL` | local SQLite | Use a hosted Postgres to keep history across reboots (see *Database*). |
| `EOD_REPORT`, `EOD_FORCE_SQUARE_OFF` | `true`, `true` | Build the daily report after each close; close paper positions still open then. |
| `LOG_RETENTION_DAYS`, `DECISION_RETENTION_DAYS` | `30`, `90` | How long log / scan-decision rows are kept (trades and reports are kept forever). |
| `HOLIDAYS_FILE` | `config/market_holidays.txt` | File of variable exchange holidays (`YYYY-MM-DD [NSE|MCX] name`). |
| `LOGS_DIR` | `logs/` | Where `system.log` is written (tests point it at a temp dir). |
| `OPTION_PREMIUM_PCT`, `OPTION_DELTA` | `1.5`, `0.5` | Assumed option premium (% of the underlying's price) and delta used to convert the underlying backtest into option P&L and costs. |

> The statutory rates are the published exchange/government charges (taken from Zerodha's public charges
> table, since Dhan's pricing page lists only brokerage, GST and the SEBI fee). Check them against your own
> Dhan contract note and override via the variables above if they differ.

Watchlists live in code: **`config/watchlist.py`** (15 stocks and 8 MCX commodities).

Stocks: RELIANCE, BHARTIARTL, HDFCBANK, ICICIBANK, SBIN, TCS, BAJFINANCE, LT, LICI, SUNPHARMA, HINDUNILVR,
TITAN, ADANIPORTS, INFY, KOTAKBANK. Commodities: CRUDEOIL, CRUDEOILM, NATURALGAS, GOLD, GOLDM, SILVER,
SILVERM, COPPER (lot multipliers come from MCX's published specs, not Dhan - verify before trusting P&L).
Use real NSE symbols (e.g. `HDFCBANK`, `LICI`).

## Deploying on Streamlit Community Cloud

1. Push the repo to GitHub; create the app with main file `quant_intelligence/app.py`.
2. App **Settings -> Secrets**:
   ```toml
   DHAN_CLIENT_ID = "your-client-id"
   DHAN_ACCESS_TOKEN = "your-token"
   ```
   Top-level secrets are exposed as environment variables, so no code change is needed.
3. **Refresh the token daily** (it expires) and reboot the app.
4. Keep the app **private** (restrict viewers) - the account, controls and kill switch are shared by
   everyone who can open it.

Cloud caveats: servers run in UTC (the app uses an IST clock internally); the disk is ephemeral, so the
SQLite database, paper account and cached candles reset on reboot; free apps **sleep after inactivity**,
which stops the background traders - "run until I stop it" cannot be guaranteed there; and Dhan generally
requires a whitelisted static IP for *order* APIs, which the cloud does not provide (data APIs are fine).
A `.devcontainer/` is included for Codespaces.

### Deploying with the research dashboard

The *Strategy Evidence* page reads the database, not the candle files, so a deployment shows the research results as
long as they are in the database it points at. The history itself (`data_cache/`, over 1 GB with the option history)
is gitignored and stays on your machine. Recommended flow:

1. Create a hosted Postgres (see *Database* below) and set `DATABASE_URL` to it in the app's secrets.
2. On your own machine, with the same `DATABASE_URL` in `.env`, run `python scripts/run_research.py NIFTY --force`.
   The evidence, status changes and research versions are written straight to Postgres, and the deployed app shows
   them. (`scripts/migrate_db.py` copies trading history only, not the research tables.)
3. Rerun it whenever you extend the history. The deployed app cannot run the long backtests itself: it has no candles.
4. Set `OPTION_HISTORY_IN_BACKTEST`, `GATED_RANKER` and `RESEARCH_SCHEDULER` on the host only when it has the data
   to back them. They all default to off.

For an always-on host, the `Dockerfile` runs the app on `$PORT` (Render, Railway, Fly.io or a VPS). Put the Dhan
credentials and `DATABASE_URL` in the host's environment variables, never in the image, and mount a volume on
`/app/data_cache` if you want the candle cache to survive restarts. Vercel cannot run Streamlit.

## Project layout

```
quant_intelligence/
  app.py                       Streamlit command center
  pages/                       01-14 dashboard pages
  config/        settings.py (env config) | watchlist.py (stocks, commodities) | credentials.py (token updates)
  reports/       performance.py (strategy analytics) | eod.py + eod_job.py (daily report) | decision_log.py | history_io.py (CSV restore) | export.py
  data/          data_manager.py (source order, cache, quality) | data_keeper.py (background refresh) | quality.py (validation/cleaning)
  data_adapters/ dhan_adapter | dhan_instrument_master (scrip master) | csv_adapter | nse_heatmap | synthetic (tests only)
  features/      feature_engine.py
  market_state/  latest feature snapshot
  regimes/       regime_engine.py
  strategies/    base_strategy.py | registry.py | 17 strategy modules
  backtesting/   engine.py (backtest, costs, walk-forward)
  analogues/     analogue_engine.py (k-NN of similar past setups)
  ranking/       ranking_engine.py (score, eligibility, NO TRADE)
  research/      pipeline.py (orchestrates data -> ranking)
  options/       chain_analytics | option_selector | premium_model
  execution/     engine.py (background runners) | multi_cycle | auto_trader | setup_detector
                 position_monitor | square_off | capital
  risk/          risk_engine.py
  brokers/       base_broker | paper_broker | dhan_broker (live) | dhan_api_client (REST)
                 dhan_rate_limit (account-wide limiter, 429 breaker) | dhan_cache (shared price/chain caches) | dhan_auth (token renew/TOTP)
  database/      db.py | models.py
  reports/       report_generator.py
  ui/            state | theme | components (status bar, KPI tiles, position cards) | charts (Plotly) | tables
                 format (Indian numbers) | position_views, chain_views (pure analytics: P&L, stats, ladder, ticket)
                 open_positions (shared live positions block) | actions (manual exit) | market_data (cached quotes)
                 multi_instrument_panel | credentials_panel
  utils/         timeutil (IST) | market_profile (NSE/MCX) | market_calendar | logging_utils
  tests/         pytest suite
scripts/         init_db.py | smoke_test.py | ui_smoke_test.py
data_cache/      (gitignored) SQLite DB, parquet candle cache, optional csv/
logs/            (gitignored) system.log (JSON lines)
```

## Database

SQLAlchemy models (`database/models.py`), SQLite by default. Tables: `instruments`,
`market_data_metadata` (provenance and quality report of every data fetch), `market_states`,
`regime_states`, `strategy_definitions`, `strategy_observations`, `backtest_runs`, `backtest_trades`,
`strategy_metrics`, `analogue_matches`, `signals`, **`orders`**, **`fills`**, **`positions`**,
**`risk_events`** (every approval and veto), `system_events`, `research_reports`, **`scan_decisions`** (what was decided on every bar), **`daily_reports`**, `vol_forecasts`, `vol_model_scores`. Columns added to models
later are migrated automatically on start (`ALTER TABLE ... ADD COLUMN`).

### Keeping the trade history permanently (Postgres)

The default database is a SQLite file on the server's disk. **Streamlit Cloud erases that disk whenever the app
reboots or sleeps**, which takes every earlier trade and report with it. To keep them, point `DATABASE_URL` at a
free hosted Postgres:

1. Create a free database at [Neon](https://neon.tech) or [Supabase](https://supabase.com) and copy its connection
   string (looks like `postgresql://user:password@host/dbname?sslmode=require`; `postgres://...` also works).
2. In your Streamlit app: *Settings -> Secrets* and add `DATABASE_URL = "postgresql://..."`. The password is a
   secret - never put it in git or chat. Reboot the app. Tables are created and upgraded automatically.
3. Check *System Logs & Health*: the Database tile should say `OK (postgresql)` and the red SQLite warning disappears.
4. Optional - bring existing history across from a local SQLite file: set `DATABASE_URL` in your shell or `.env`
   to the Postgres URL, then `python scripts/migrate_db.py` (dry run) and `python scripts/migrate_db.py --apply`.
   Safe to repeat; it never duplicates and skips the fake test trades.

Free tiers are about 0.5 GB, so old log rows are trimmed daily (`LOG_RETENTION_DAYS` 30, `DECISION_RETENTION_DAYS`
90); trades, orders, fills and reports are never trimmed. The Postgres driver and schema upgrade are covered by unit
tests, but the first real connection can only be confirmed on your own database.

**Restoring from files:** *Positions & Orders -> Trade history* has a **Download positions (CSV)** button and a
**Restore earlier trades from a CSV** importer (matched on position id, so importing twice never duplicates).
A position that was still open in the file comes back as STALE (no exit, excluded from statistics).

### Daily report and decision log

Every closed bar the system records what it decided per instrument (`scan_decisions`: no-trade reason, setup
waiting/triggered, risk veto, rejected, filled), and every trade stores what the ranker expected at entry
(`expected_r`, `confidence`, `regime`). After each close the data keeper builds the day's report (NSE about 15:40 IST,
MCX just after midnight for the session that ended) into `daily_reports`, first settling any paper position still open
(`EOD_FORCED` exit at the latest traded price; with no token it stays open and the report lists it). Switch off with
`EOD_REPORT=false` / `EOD_FORCE_SQUARE_OFF=false`.

**How to read the verdicts.** R = net P&L after charges divided by the premium risked at the stop, so trades of
different size compare. A strategy is only called good or bad with at least 30 closed trades; "statistically
significant" means the mean R is at least two standard errors from zero. With fewer trades the report says
*too few trades to conclude* however good the average looks - early paper results are mostly noise. Positions
left open when the app slept (STALE) have no result and are reported separately.



## Testing

```bash
python -m pytest -q          # 300+ tests, ~1 minute, no network
python scripts/ui_smoke_test.py   # runs every page headlessly
```

Tests cover the data-quality gate, timezone alignment, feature no-look-ahead, strategies, backtester,
analogues, ranking, the risk engine, the paper broker, option selection and the premium model, Dhan client
request shapes/retry/caching (HTTP is mocked), capital accounting, the background engine (start/stop,
failure isolation, market-hours and kill-switch behaviour), square-off and restart recovery. Tests use a
temporary database and log directory and never write to your real ones, nor start the background keepers.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Dhan API error (401) ... DH-901 Invalid_Authentication` | Access token expired (about 24 h) or wrong. Generate a new one and update the sidebar / Secrets. |
| `Dhan API error (429) ... 805 Too many requests` | Dhan's per-account rate limit. The app spaces, caches and retries calls and pauses briefly if it persists (see *Health* page / sidebar banner). Usually another app copy, tab or script is using the same account - run **one** instance. |
| "Dhan rejected your credentials (401)" / "Authentication Failed" | The access token expired or is wrong. Requests are paused for 5 min or until you enter a new token under **API Keys**. |
| `DataUnavailableError: No real market data ...` | No CSV and Dhan failed or isn't configured; the message lists the reason. |
| Data quality `FAIL` / `DEGRADED` | Read the issues in the log: stale data (market closed), gaps, bad bars, or misaligned timezone. The system stays in NO TRADE. |
| Everything is `NO TRADE` | Often correct: no strategy clears the edge/confidence bar, or data isn't `OK`. See Known limitations. |
| Chain error for one instrument | Check `logs/system.log`; usually rate limit or an expired token. |
| Data feed shows "N stale" | Read the reason on the *Health* page. Usually an expired Dhan token (see *Token lifecycle*), or a holiday not yet in `market_holidays.txt`. The keeper retries every minute and recovers by itself once the token is valid. |
| NIFTY says NO TRADE "not distinguishable" | The top strategies are statistically tied. With `TIE_BREAK_PAPER` on the paper trader takes the top one (tagged TIE-BREAK, half size); with it off it stays out. |
| Database is gigabytes / app is slow | Older versions saved a full backtest per strategy on every scan (~700k rows). Scans no longer do; remove the old rows with `python scripts/prune_research_tables.py` (dry run) then `--apply` (backs up first). Orders, positions and fills are never touched. |
| Data feed says "paused - token expired" | The keeper stops sending requests while the token is expired/rejected (no log spam) and resumes by itself when you save a new one. |
| Fake "Momentum" trades at premium ~100 in Positions | Left by early test runs. `python scripts/purge_test_trades.py` lists them (dry run); `--apply` backs up the DB then deletes them. |
| Trades missing after a restart | Paper state is in memory; today's book is restored from the database, older open rows are marked `STALE`. |
| Times look 5.5 h off | Server clock is UTC; the app uses IST internally - restart on the latest version. |

## Extending it

- **Add a strategy:** subclass `BaseStrategy` (implement `generate_historical_setups` and `check_setup`),
  then register it in `strategies/registry.py`. The backtester, ranking engine and UI pick it up with no
  other changes.
- **Change the watchlist:** edit `config/watchlist.py` (stocks must be NSE F&O symbols; commodities
  need their lot multiplier).
- **Add a data source:** implement `data_adapters/base.py::DataAdapter` and add it to `DataManager`.
- **Tune risk:** use the `MAX_*` environment variables - the risk engine reads them at start-up.

## Known limitations (read this before trusting a signal)

- **No proven edge.** The strategies are well-known indicator setups. Nothing here has been shown to be
  profitable after costs: on ~60 days of 5-minute data the strategies are roughly breakeven *before* costs
  and slightly negative after realistic option costs. An earlier walk-forward check of the original ranker
  (17 instruments, ~260 trades) found **no measurable advantage over picking a strategy at random**; that
  check used an older, inflated cost model, and re-measuring the current ranker with `evaluate_ranker` is the
  way to judge it. Treat output as research, run
  extended paper trading, and never size real money from it without your own validation.
- **Costs are modelled, not measured.** The backtest has no historical option prices, so premium is assumed
  to be a fixed % of the underlying and to move by a fixed delta (`OPTION_PREMIUM_PCT`, `OPTION_DELTA`);
  real premiums, spreads, theta and IV differ by instrument, strike and time to expiry. Statutory rates come
  from published tables - verify them against your contract note. With these costs, a net-of-cost ranker may
  say NO TRADE most of the time when no strategy shows an edge; that is intended.
- **Researched on the underlying, traded in premium.** Edge is measured on index/stock price moves;
  execution is option premium, which also moves with theta, IV and bid-ask spread. The premium model is a
  delta approximation.
- **Chain-aware strategies cannot be ranked.** PCR, OI-buildup, IV-crush and unusual-volume strategies have
  no historical setups, so the ranker never selects them.
- **Index volume is unavailable** from Dhan, so volume-based features and strategies are skipped for
  NIFTY/BANKNIFTY.
- **Ranking is statistically thin.** Analogues are the 30 nearest setups, mostly from the same sessions
  (autocorrelated), and the winner is picked from many strategies on the same data used to estimate its
  edge - so the optimism in "best of N" is not corrected.
- **Tie-break trades have no proven edge.** When the two best strategies cannot be told apart (score gap < 0.08 or
  z < 1.0) the ranker used to say NO TRADE. By choice (`TIE_BREAK_PAPER`, `TIE_BREAK_LIVE`, both on) it now picks
  the top one if it is still *eligible* (positive shrunk edge, adequate confidence, data OK), tags the order
  `TIE-BREAK` and sizes it at `TIE_BREAK_SIZE_FACTOR` (0.5) of normal. Those are by construction the cases where
  the statistics could not separate the candidates, so expect them to be roughly breakeven-to-negative after costs
  (see the evaluator numbers below). Live trading is still manual (double gate, confirmation checkbox, risk engine
  unchanged); the Live page shows a red TIE-BREAK banner. Turn it off with `TIE_BREAK_LIVE=false` /
  `TIE_BREAK_PAPER=false` (data not OK, nothing eligible or an ineligible leader stay NO TRADE regardless).
  Measured with `python -m quant_intelligence.ranking.evaluate_ranker NIFTY BANKNIFTY --max-points 400 [--tie-break]`
  (~60 days of 5-minute data, 2 Oct 2026):

  | | Trades | Mean net R / trade | Win rate | Random-pick mean | Take-every-signal |
  |---|---|---|---|---|---|
  | Tie-break off | 23 | -0.32 | 26% | -0.26 | -0.14 |
  | Tie-break on | 43 | -0.22 | 30% | -0.22 | -0.14 |

  Tie-break nearly doubles the trades but the lift over a random pick is ~0 (95% interval -0.005 to +0.004) and
  the mean net R's 95% interval (-0.55 to +0.14) includes a loss: more trades, not better ones, and each pays
  real costs. The samples are small, so do not read anything into the NIFTY/BANKNIFTY difference. Re-run the
  command above on your own data before relying on it, and keep `TIE_BREAK_LIVE` off until paper trading of the
  tagged trades shows they are not losing.
- **Tighter stops did not improve results.** `python scripts/evaluate_stop_scales.py` (all 17 strategies on 25 cached
  instruments, ~60 days of 5-minute data, Black-Scholes option premium, one lot per trade, 7 Oct 2026):

  | `STOP_ATR_SCALE` | Trades | Win rate | Mean net R | Median stop (% of premium) | Median one-lot loss | Trades that fit Rs 1,800 |
  |---|---|---|---|---|---|---|
  | 1.0 (default) | 22,867 | 34.6% | -0.44 | 4.0% | Rs 741 | 83% |
  | 0.75 | 24,768 | 34.3% | -0.49 | 3.3% | Rs 594 | 85% |
  | 0.5 | 26,267 | 34.9% | -0.58 | 2.4% | Rs 441 | 87% |

  A tighter stop raises the share of trades one lot can afford only slightly, leaves the win rate unchanged and makes
  net R worse (more small losses, and the fixed brokerage is a bigger share of a smaller risk). The intervals are
  overstated because strategies and bars overlap, and these are all signals, not the ranker's picks, so treat the
  direction as the finding, not the decimals. The loss per stopped trade is therefore reduced through position size
  (`MAX_RISK_PER_TRADE_PCT`), not by tightening stops. GOLD and SILVER (lots of 100 and 30) almost never fit the
  Rs 1,800 budget; their mini contracts (GOLDM, SILVERM) do.
- **Paper != live.** Paper fills use simple slippage on the last price; real fills depend on liquidity and
  spread. Sold-option margin is not modelled. Only holidays listed in `market_holidays.txt` (plus the fixed-date ones) are modelled.
- **Live trading is unproven** against a funded account.
- **Paper state is in memory** (restored from the database on restart for the current day only).

## Research rigor (evidence-gated pipeline, in progress)

- **History and data quality:** `python scripts/backfill_history.py NIFTY --years 5` extends the candle cache backward
  and forward and audits it (`data/quality_gates.py`: session completeness, gaps, outliers, dead-volume runs, plus a
  dataset fingerprint). It records how far back Dhan actually serves. Needs a valid Dhan token.
- **Rules:** `strategies/rules.py` states each strategy's entry, exit, stop, target, session, re-entry and parameter
  ranges as the code implements them; `tests/test_strategy_rules.py` keeps them in step with the registry.
- **No look-ahead:** `tests/test_no_lookahead.py` truncates the data and checks every price-action strategy's signals
  (and the features) are unchanged by later bars. A canary strategy that peeks at the future must fail it.
- **Execution timing:** by default a backtest fills at the signal bar's own close, a price that was not achievable
  (legacy, kept so past results stay comparable). `EXECUTION_TIMING=next_open` fills at the next bar's open, counts the
  fill bar's range, flattens at the session's last bar and skips signals with no honest fill (reported as
  `n_unexecutable`). It changes more than the fill price, so compare results rather than assuming it is pessimistic.
  Use it for any research claim.
- **Reproducibility:** every run carries a manifest (`backtesting/manifest.py`: dataset fingerprint, strategy version,
  parameters, execution and cost assumptions, engine version, `config_hash`), stored in `backtest_runs.manifest`.
  Metrics report gross (`gross_pnl_total`) and net (`net_pnl_total`) separately.

- **Statistical validation** (`quant_intelligence/validation/`): `metrics.py` (win rate, average win/loss, net expectancy
  `E = pW - (1-p)L - C`, profit factor, drawdown, exposure, frequency, volatility, monthly variability),
  `uncertainty.py` (expectancy interval = the wider of a stationary block bootstrap and an effective-sample-size
  interval; Wilson win-rate interval; effective N discounted for same-session clustering and serial correlation) and
  `evidence.py`. A backtest lands in one of six states, never a bare score: INSUFFICIENT DATA, INSUFFICIENT SAMPLE,
  VALIDATION FAILED, RESEARCH QUALIFIED, ROBUSTNESS TESTING, VALIDATED FOR RESEARCH. The labels describe evidence under a
  documented policy (`EvidencePolicy` in `config/settings.py`, overridable with `EVIDENCE_*` variables) and never
  promise profit. `validation.report.validate_backtest(result, ohlcv, quantity)` returns metrics, uncertainty and state.

- **Splits, walk-forward, holdout, robustness:** `validation.suite.run_validation_suite(StrategyClass, ohlcv, features,
  instrument)` cuts history chronologically by whole sessions into development / validation / holdout (60/20/20, with
  purge and embargo sessions at each boundary; `SplitPolicy`, `SPLIT_*` variables), judges the in-sample gates on
  development and out-of-sample on validation, and runs a session-aligned walk-forward (`walk_forward.py`; no
  parameter is ever fitted, so folds are consecutive cold runs and what they measure is consistency over time).
  The five robustness tests (`robustness.py`, pass marks in `RobustnessPolicy`) run only once the earlier gates have
  passed: parameter sensitivity (neighbours of the defaults, never the best in-sample setting), regime sensitivity
  (labels from information at the signal bar; the strategy x regime table is appended to `strategy_regime_stats`),
  cost stress (slippage scaled up; the break-even multiple is exact), period consistency (monthly) and sequence
  analysis (block-bootstrapped drawdown). A test with too little data is "not run", never a pass.
- **The holdout is for one final look.** It is only evaluated (`include_holdout=True`) for a strategy that would
  otherwise be VALIDATED FOR RESEARCH, and every look is appended to `holdout_ledger`. A holdout seen by too many
  strategies, or twice by the same configuration, is reported as consumed. Fresh data gives a fresh holdout.
- **Gated ranker** (`GATED_RANKER=true`, default off; `ranking/gated_ranker.py`). Evidence comes before score. The suite
  stores each verdict in `strategy_evidence` (`run_validation_suite(..., persist=True)`). When the flag is on, a strategy
  can be selected only if its latest stored evidence passes every gate: live data quality OK, evidence current (same
  strategy version, within `GATED_MAX_EVIDENCE_AGE_DAYS`), state at least `GATED_MIN_STATE`, net-edge interval above
  zero, a positive out-of-sample result, drawdown within policy and no failed robustness test. Eligible strategies are
  ordered in tiers (evidence state, then bands of out-of-sample lower bound, walk-forward consistency, parameter
  stability and cost sensitivity), never a weighted sum. The legacy ranker still supplies "does it fit right now" and
  breaks ties inside the top tier. Each strategy gets separate `research_status`, `compatibility`, `setup_status` and
  `final_status` fields, and the best final status is `SETUP_VALID_PENDING_RISK`: the ranker never grants permission, the
  risk engine still evaluates every order. **With no stored evidence every strategy is ineligible and the answer is NO
  TRADE**, so run the suite with `persist=True` before switching the flag on.
- **Research scheduler and reports** (`RESEARCH_SCHEDULER=true` starts a background thread; or run
  `python scripts/run_research.py NIFTY` by hand, `--status` to look without running). Per instrument and timeframe it
  loads the cached candles, applies the quality gate (a failed or missing history is logged `AWAITING_DATA`, nothing is
  run), re-validates only the strategies that are stale (never run at this strategy version, run under different
  thresholds, or `RESEARCH_RERUN_AFTER_SESSIONS` = 5 new sessions of data since), always with next-open fills and never
  touching the holdout, and defers while the market is open. Each verdict then meets the acceptance rules
  (`research/acceptance.py`): **a fall is accepted at once** (DEMOTION), a drop of more than 0.05 R in the out-of-sample
  lower bound is flagged (DETERIORATION), and **a higher state is accepted only after 20 new sessions of data under
  unchanged thresholds**, otherwise it is logged PROMOTION_HELD and the accepted verdict stands. Improvement within a
  state is a REFRESH and never a promotion. The gated ranker reads only accepted verdicts. Everything is append-only:
  `research_versions` (the accepted leaderboard each time it changes), `status_changes` (with reasons),
  `research_run_log` (every attempt: completed, failed, awaiting data, deferred), plus a Markdown/JSON run report per
  batch in `research_reports`. `prune_research_tables.py` lists these tables and never deletes from them.
- **Strategy Evidence page** (`pages/17_Strategy_Evidence.py`; the older "Strategy Intelligence" page is the analogue
  drill-down). Four tabs over the stored, accepted evidence, with the logic in `research/dashboard_data.py` and the
  drawing in `ui/evidence_views.py`: a **leaderboard** (state, net R with its interval, profit factor, drawdown,
  out-of-sample, walk-forward, robustness, regime fit, setup, final status, evidence age; with a banner that sorting is not
  a performance guarantee), **research detail** per strategy (equity and drawdown with the out-of-sample stretch marked,
  trade distribution, monthly and per-condition breakdowns, parameter-sensitivity and cost-stress charts, walk-forward
  folds, robustness results, trade records, rules and known limitations), a **backtest control center** (completed,
  failed and awaiting-data runs, current research version, last success, data coverage and quality, held promotions,
  status changes, and buttons to run stale research or re-run everything in the background) and a **ranking
  explanation** (research rank with the strategies just above and below, gates passed and failed, fit with the current
  market, setup, and what remains uncertain). Curves and tables come from a compact detail block stored with each
  verdict (`validation/detail.py`); where a piece was not computed the page says so. The Backtest Lab now shows each run's
  manifest.
- **Option data from Dhan** (`python scripts/backfill_options.py NIFTY --days 365`; needs a valid token and a data plan
  with expired-options data). It uses `POST /v2/charts/rollingoption` (30 days per request, ATM-relative strikes, with
  OHLC, volume, IV, OI, strike and spot) and stores one parquet per expiry flag, option type and strike offset under
  `data_cache/options_history/<SYMBOL>/`. The response handling follows Dhan's documentation and has **not been checked
  against a live response from this account yet**; the parser refuses shapes it does not recognise. Two uses:
  - `OPTION_HISTORY_IN_BACKTEST=true` prices each bought ATM option from the stored history (entry at the fill bar's open
    under `next_open`, exit at the close of the same strike; delta read from the neighbouring strikes). Trades it cannot
    price keep the modelled/assumed premium and are counted apart (`n_real_premium`, `n_modelled_premium`,
    `n_assumed_premium`); the manifest records the option-data fingerprint. No bid/ask in the data, so spread stays assumed.
  - `options/historical_chain.py` rebuilds causal near-the-money `ChainSnapshot`s so PCR Extreme, OI Buildup and Unusual
    Volume Fade can be replayed (`run_validation_suite(..., chain_history=HistoricalChain(history))`). These are proxies:
    only the stored strikes, so PCR is a near-the-money ratio, not the whole-chain one the thresholds were written for.
    The result is flagged `proxy_data` and can never reach VALIDATED FOR RESEARCH. IV Crush Premium Writer sells premium
    and has no sell-side backtest model, so it stays INSUFFICIENT DATA.
  - `CHAIN_RECORDER=true` saves the live full chain (every strike) as it is fetched, at most once per
    `CHAIN_RECORD_INTERVAL_SEC` per underlying, to `data_cache/chain_snapshots/`. It makes no extra Dhan requests.
    This is the only route to a true whole-chain history, and it accumulates from the day it is switched on.

## Disclaimer

This software is provided for research and educational purposes. Trading derivatives involves substantial
risk of loss. Nothing here is investment advice, and the authors accept no liability for any losses.
