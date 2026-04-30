"""
Deriv ExpiryRange Quant Bot  -  single file, self-calibrating
Symbol : 1HZ10V
Contract: EXPIRYRANGE (Ends In)
Expiry  : 2 minutes
Barrier : +/-1.9

FIXES IN THIS VERSION
─────────────────────
1. SETTLEMENT DEADLOCK  (critical)
   _execute() no longer sleeps inside the tick-callback coroutine.
   Settlement is dispatched as a separate asyncio Task so tick processing
   continues uninterrupted during the 2-minute contract window.
   The in-trade lock is set immediately on buy confirmation and released
   only after settlement confirms, preserving all risk-management semantics.

2. IN-TRADE DOUBLE-CHECK REMOVED
   on_tick() previously checked risk._in_trade directly (private attr) and
   then called can_trade() which checks it again.  The direct check was
   removed; can_trade() is the single authoritative gate.  Drought counter
   (_ticks_since_trade) is now only incremented when the bot is NOT in a
   trade, so an active 2-minute contract does not inflate the deadlock timer.

3. SETTLE LOCK-LEAK GUARD
   _settle() wraps its entire polling loop in a try/finally that always
   calls release_trade_lock() if settlement could not be confirmed, even
   if an unexpected exception occurs mid-poll.  Previously an exception
   during polling would leave _in_trade=True permanently.

4. asyncio.get_event_loop() REPLACED
   All _rpc() calls now use asyncio.get_running_loop() which is correct
   for Python 3.10+ and avoids DeprecationWarning / RuntimeError in newer
   runtimes.

5. DUPLICATE _save() ON SHUTDOWN REMOVED
   shutdown() saves state and sets _alive=False.  _connect_and_run() no
   longer calls _save() after disconnect to avoid the double-write race.

6. SKIP SUMMARY DECOUPLED FROM SKIP LOG RATE-LIMITER
   The periodic skip summary previously only printed if the rate-limiter
   happened to fire on the right tick.  It now has its own counter and
   fires independently every skip_summary_every ticks after warmup.

Run:
    export DERIV_API_TOKEN=your_token
    export DERIV_APP_ID=1089          # optional, default 1089
    python deriv_bot.py

Backtest (no API needed):
    python deriv_bot.py --backtest
"""

import asyncio
import csv
import json
import logging
import math
import os
import pickle
import random
import signal
import sys
import time
from collections import deque, Counter
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Callable, Dict, List, Optional, Tuple

import websockets


# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"bot_{datetime.now().strftime('%Y%m%d')}.log"),
    ],
)
log = logging.getLogger("bot")


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

@dataclass
class Config:
    # Deriv API
    api_token: str = field(default_factory=lambda: os.getenv("DERIV_API_TOKEN", ""))
    app_id:    str = field(default_factory=lambda: os.getenv("DERIV_APP_ID", "1089"))
    api_url:   str = "wss://ws.binaryws.com/websockets/v3"

    # Contract
    symbol:        str   = "R_10"
    expiry_min:    int   = 2
    barrier:       float = 0.9
    contract_type: str   = "EXPIRYRANGE"
    currency:      str   = "USD"
    payout_ratio:  float = 0.49   # actual observed payout ~$0.17 on $0.35 stake

    # Stake
    base_stake: float = 0.35
    min_stake:  float = 0.35
    max_stake:  float = 10.0
    kelly_frac: float = 0.25

    # Warm-up & windows
    warmup_ticks:    int = 300
    vol_window:      int = 150
    range_window:    int = 120
    spike_window:    int = 20
    cal_history:     int = 500
    t_ticks:         int = 120

    # Auto-threshold percentiles
    vol_percentile:   float = 38.0
    range_percentile: float = 30.0
    ema_percentile:   float = 30.0

    # Absolute EMA floor — kept because EMA distance can genuinely collapse to
    # near-zero in a tight range and a pure percentile would let everything through.
    floor_ema_abs: float = 0.05

    # Balance guard
    balance_guard_multiples: int = 5   # need balance >= min_stake × this

    # Spike filter
    spike_k: float = 3.0

    # Z-score time-aware floor
    z_coverage_factor:       float = 1.0
    zscore_floor_window:     int   = 300
    zscore_floor_percentile: float = 55.0
    zscore_floor_min:        float = 0.55
    zscore_floor_max:        float = 1.80

    # Deadlock detector
    deadlock_scan_limit:    int   = 2000   # ticks WITHOUT a trade → trigger
    deadlock_widen_factor:  float = 1.35
    deadlock_relief_ticks:  int   = 500

    # Bayes threshold bounds
    bayes_min_threshold:      float = 0.38
    bayes_max_threshold:      float = 0.75
    bayes_explore_threshold:  float = 0.35
    bayes_bootstrap_n:        int   = 20
    bayes_learn_n:            int   = 100

    # Risk
    cooldown_after_loss:      int   = 60
    max_consecutive_losses:   int   = 3
    max_daily_loss_pct:       float = 0.15

    # Martingale
    marti_factor:    float = 3.1
    marti_kick_in:   int   = 2
    marti_max_steps: int   = 4

    # Settlement verification
    settle_wait_extra:    int = 10    # seconds to wait after nominal expiry
    settle_poll_interval: int = 5     # seconds between polls
    settle_poll_attempts: int = 12    # max polls (12 × 5s = 60s extra)

    # Skip logging
    skip_log_interval:  float = 30.0
    skip_summary_every: int   = 150

    # Persistence
    state_file:   str = "bot_state.pkl"
    history_file: str = "trade_history.csv"


# -----------------------------------------------------------------------------
# PERCENTILE HELPER
# -----------------------------------------------------------------------------

def percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    s   = sorted(data)
    n   = len(s)
    idx = pct / 100.0 * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def mean_std(data: List[float]) -> Tuple[float, float]:
    if len(data) < 2:
        return (data[0] if data else 0.0), 0.0
    mu  = sum(data) / len(data)
    var = sum((x - mu) ** 2 for x in data) / len(data)
    return mu, math.sqrt(var)


# -----------------------------------------------------------------------------
# BAYESIAN MODEL
# -----------------------------------------------------------------------------

class BayesModel:

    REGIMES = ("low", "medium", "high")

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Uniform [1,1] prior — real outcomes move the posterior fast.
        self._bb: Dict[str, List[float]] = {r: [1.0, 1.0] for r in self.REGIMES}
        self._w  = [0.0] * 5
        self._b  = 0.0
        self._lr = cfg.bayes_min_threshold * 0.08
        self._l2 = 0.001
        self._n  = 0

    def adaptive_min_threshold(self) -> float:
        """
        Sliding Bayes gate floor based on number of confirmed trades.
        - Exploration (n < bootstrap_n): use bayes_explore_threshold.
        - Learning    (bootstrap_n ≤ n < learn_n): linearly ramp to bayes_min_threshold.
        - Mature      (n ≥ learn_n): use bayes_min_threshold.
        """
        n   = self._n
        cfg = self.cfg
        if n < cfg.bayes_bootstrap_n:
            return cfg.bayes_explore_threshold
        if n < cfg.bayes_learn_n:
            t = (n - cfg.bayes_bootstrap_n) / (cfg.bayes_learn_n - cfg.bayes_bootstrap_n)
            return cfg.bayes_explore_threshold + t * (cfg.bayes_min_threshold - cfg.bayes_explore_threshold)
        return cfg.bayes_min_threshold

    def predict(self, fv: List[float], regime: str) -> Tuple[float, float]:
        a, b    = self._bb[regime]
        p_bb    = a / (a + b)
        var_bb  = (a * b) / ((a + b) ** 2 * (a + b + 1))
        sd_bb   = math.sqrt(var_bb)
        p_lr    = self._sigmoid(self._b + sum(wi * fi for wi, fi in zip(self._w, fv)))
        w_lr    = min(0.7, self._n / 200)
        p_final = (1 - w_lr) * p_bb + w_lr * p_lr
        threshold = p_bb - 0.5 * sd_bb
        min_th    = self.adaptive_min_threshold()
        threshold = max(min_th, min(self.cfg.bayes_max_threshold, threshold))
        return p_final, threshold

    def update(self, fv: List[float], regime: str, won: bool):
        a, b = self._bb[regime]
        self._bb[regime] = [a + (1 if won else 0), b + (0 if won else 1)]
        y   = 1.0 if won else 0.0
        p   = self._sigmoid(self._b + sum(wi * fi for wi, fi in zip(self._w, fv)))
        err = p - y
        self._b -= self._lr * err
        for i, fi in enumerate(fv):
            self._w[i] -= self._lr * (err * fi + self._l2 * self._w[i])
        self._n += 1
        log.info(
            f"Bayes update | regime={regime} won={won} "
            f"p_bb={self._bb[regime][0]/(sum(self._bb[regime])):.3f} n={self._n}"
        )

    def threshold_for(self, regime: str) -> float:
        a, b   = self._bb[regime]
        p_bb   = a / (a + b)
        var_bb = (a * b) / ((a + b) ** 2 * (a + b + 1))
        sd_bb  = math.sqrt(var_bb)
        t      = p_bb - 0.5 * sd_bb
        return max(self.adaptive_min_threshold(), min(self.cfg.bayes_max_threshold, t))

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"bb": self._bb, "w": self._w, "b": self._b, "n": self._n}, f)

    def load(self, path: str):
        try:
            with open(path, "rb") as f:
                s = pickle.load(f)
            self._bb = s["bb"]
            self._w  = s["w"]
            self._b  = s["b"]
            self._n  = s["n"]
            log.info(f"Model loaded from {path} | trades={self._n}")
        except FileNotFoundError:
            log.info("No saved model — starting fresh.")

    def summary(self) -> str:
        min_th = self.adaptive_min_threshold()
        phase  = ("explore" if self._n < self.cfg.bayes_bootstrap_n
                  else "learning" if self._n < self.cfg.bayes_learn_n
                  else "mature")
        lines  = [f"Bayesian model (n={self._n} phase={phase} min_th={min_th:.3f}):"]
        for r in self.REGIMES:
            a, b = self._bb[r]
            n    = int(a + b - 2)
            lines.append(
                f"  {r:7s}: p={a/(a+b):.3f}  threshold={self.threshold_for(r):.3f}"
                f"  N={max(0,n)}"
            )
        return "\n".join(lines)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-max(-30, min(30, x))))


# -----------------------------------------------------------------------------
# FEATURES
# -----------------------------------------------------------------------------

@dataclass
class Features:
    tick:          int
    price:         float
    sigma:         float
    sigma_sqrt_T:  float
    sigma_price:   float
    mu:            float
    range_width:   float
    ema_fast:      float
    ema_slow:      float
    ema_distance:  float
    zscore:        float
    max_jump:      float
    regime:        str
    fv:            List[float]


# -----------------------------------------------------------------------------
# TICK BUFFER + FEATURE ENGINE
# -----------------------------------------------------------------------------

class TickBuffer:

    def __init__(self, cfg: Config):
        self.cfg     = cfg
        maxlen       = max(cfg.vol_window, cfg.range_window, cfg.spike_window, 500)
        self._prices:           deque = deque(maxlen=maxlen)
        self._ema_fast:         Optional[float] = None
        self._ema_slow:         Optional[float] = None
        self._tick              = 0

        self._hist_sigma_sqrtT: deque = deque(maxlen=cfg.cal_history)
        self._hist_range:       deque = deque(maxlen=cfg.cal_history)
        self._hist_ema_dist:    deque = deque(maxlen=cfg.cal_history)
        self._hist_jumps:       deque = deque(maxlen=cfg.cal_history)
        self._hist_zscore_abs:  deque = deque(maxlen=cfg.zscore_floor_window)

    def push(self, price: float) -> Optional[Features]:
        self._prices.append(price)
        self._tick += 1
        self._update_emas(price)
        if self._tick < self.cfg.warmup_ticks:
            return None
        return self._compute(price)

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def is_warm(self) -> bool:
        return self._tick >= self.cfg.warmup_ticks

    # ── Thresholds — pure percentile, no live-value chase ─────────────────────

    def vol_threshold(self) -> float:
        return max(percentile(list(self._hist_sigma_sqrtT), self.cfg.vol_percentile), 1e-6)

    def range_threshold(self) -> float:
        return max(percentile(list(self._hist_range), self.cfg.range_percentile), 1e-6)

    def ema_threshold(self) -> float:
        pct_val = percentile(list(self._hist_ema_dist), self.cfg.ema_percentile)
        return max(pct_val, self.cfg.floor_ema_abs)

    def spike_threshold(self) -> float:
        if not self._hist_jumps:
            return 0.4
        mu, sd = mean_std(list(self._hist_jumps))
        return mu + self.cfg.spike_k * sd

    def zscore_limit(self, sigma_price: float) -> float:
        """
        Time-aware z-score ceiling — min(coverage_cap, distribution_floor).
        Coverage cap  = z_coverage_factor × (barrier / sigma_price)
        Distribution  = zscore_floor_percentile of recent |Z| history.
        Result clamped to [zscore_floor_min, zscore_floor_max].
        """
        cfg = self.cfg
        coverage_cap = (
            cfg.z_coverage_factor * (cfg.barrier / sigma_price)
            if sigma_price > 0 else cfg.zscore_floor_max
        )
        if len(self._hist_zscore_abs) >= 30:
            dist_floor = percentile(list(self._hist_zscore_abs), cfg.zscore_floor_percentile)
        else:
            dist_floor = cfg.zscore_floor_max  # cold start: don't block

        raw = min(coverage_cap, dist_floor)
        return max(cfg.zscore_floor_min, min(cfg.zscore_floor_max, raw))

    # ── Internal ───────────────────────────────────────────────────────────────

    def _update_emas(self, price: float):
        kf = 2 / 11   # EMA-10
        ks = 2 / 31   # EMA-30
        if self._ema_fast is None:
            self._ema_fast = price
            self._ema_slow = price
        else:
            self._ema_fast = price * kf + self._ema_fast * (1 - kf)
            self._ema_slow = price * ks + self._ema_slow * (1 - ks)

    def _compute(self, price: float) -> Features:
        prices = list(self._prices)

        vol_w   = prices[-self.cfg.vol_window:]
        returns = [vol_w[i] - vol_w[i-1] for i in range(1, len(vol_w))]
        _, sigma = mean_std(returns) if len(returns) > 1 else (0.0, 1e-9)
        sigma        = max(sigma, 1e-9)
        sigma_sqrt_T = sigma * math.sqrt(self.cfg.t_ticks)

        mu          = sum(vol_w) / len(vol_w)
        sigma_price = math.sqrt(
            sum((x - mu) ** 2 for x in vol_w) / len(vol_w)
        ) if len(vol_w) > 1 else 1e-9
        sigma_price = max(sigma_price, 1e-9)

        rng_w       = prices[-self.cfg.range_window:]
        range_width = max(rng_w) - min(rng_w)

        ema_distance = abs((self._ema_fast or mu) - (self._ema_slow or mu))
        zscore       = (price - mu) / sigma_price

        spk_w    = prices[-self.cfg.spike_window:]
        jumps    = [abs(spk_w[i] - spk_w[i-1]) for i in range(1, len(spk_w))]
        max_jump = max(jumps) if jumps else 0.0

        vol_t  = self.vol_threshold() or 1.2
        if sigma_sqrt_T < vol_t * 0.6:
            regime = "low"
        elif sigma_sqrt_T < vol_t:
            regime = "medium"
        else:
            regime = "high"

        self._hist_sigma_sqrtT.append(sigma_sqrt_T)
        self._hist_range.append(range_width)
        self._hist_ema_dist.append(ema_distance)
        if jumps:
            self._hist_jumps.extend(jumps)
        self._hist_zscore_abs.append(abs(zscore))

        fv = [
            min(sigma_sqrt_T / 3.0, 1.0),
            min(range_width  / (self.cfg.barrier * 2), 1.0),
            min(ema_distance / 0.5, 1.0),
            min(abs(zscore)  / 2.0, 1.0),
            min(max_jump     / 1.0, 1.0),
        ]

        return Features(
            tick=self._tick, price=price,
            sigma=sigma, sigma_sqrt_T=sigma_sqrt_T, sigma_price=sigma_price, mu=mu,
            range_width=range_width,
            ema_fast=self._ema_fast or mu, ema_slow=self._ema_slow or mu,
            ema_distance=ema_distance,
            zscore=zscore, max_jump=max_jump,
            regime=regime, fv=fv,
        )


# -----------------------------------------------------------------------------
# GATE CHAIN
# -----------------------------------------------------------------------------

@dataclass
class Decision:
    trade:      bool
    stake:      float
    p_win:      float
    score:      float
    reason:     str
    gate:       str
    margin:     float
    thresholds: Dict


class GateChain:

    def __init__(self, cfg: Config, buf: TickBuffer, bayes: BayesModel):
        self.cfg   = cfg
        self.buf   = buf
        self.bayes = bayes

    def evaluate(self, f: Features, balance: float,
                 risk: Optional["RiskManager"] = None,
                 widen_factor: float = 1.0) -> Decision:
        th_vol   = self.buf.vol_threshold()   * widen_factor
        th_range = self.buf.range_threshold() * widen_factor
        th_ema   = self.buf.ema_threshold()   * widen_factor
        th_spike = self.buf.spike_threshold() * widen_factor
        th_z     = self.buf.zscore_limit(f.sigma_price) * widen_factor
        _, th_bayes = self.bayes.predict(f.fv, f.regime)

        thresholds = {
            "vol":   round(th_vol,   4),
            "range": round(th_range, 4),
            "ema":   round(th_ema,   4),
            "spike": round(th_spike, 4),
            "z":     round(th_z,     4),
            "bayes": round(th_bayes, 4),
        }

        def reject(gate, reason, margin):
            return Decision(trade=False, stake=0, p_win=0, score=0,
                            reason=reason, gate=gate, margin=margin,
                            thresholds=thresholds)

        # Gate 1: Volatility
        if th_vol == 0 or f.sigma_sqrt_T >= th_vol:
            margin = th_vol - f.sigma_sqrt_T
            return reject("vol",
                f"sig*sqrtT={f.sigma_sqrt_T:.4f} >= th={th_vol:.4f} "
                f"(over by {-margin:.4f})", margin)

        # Gate 2: Range
        if th_range == 0 or f.range_width >= th_range:
            margin = th_range - f.range_width
            return reject("range",
                f"range={f.range_width:.4f} >= th={th_range:.4f} "
                f"(over by {-margin:.4f})", margin)

        # Gate 3: EMA compression
        if th_ema == 0 or f.ema_distance >= th_ema:
            margin = th_ema - f.ema_distance
            return reject("ema",
                f"ema_dist={f.ema_distance:.4f} >= th={th_ema:.4f} "
                f"(over by {-margin:.4f})", margin)

        # Gate 4: Z-score
        if abs(f.zscore) >= th_z:
            margin = th_z - abs(f.zscore)
            return reject("zscore",
                f"|Z|={abs(f.zscore):.4f} >= th={th_z:.4f} "
                f"(over by {-margin:.4f})", margin)

        # Gate 5: Spike
        if f.max_jump >= th_spike:
            margin = th_spike - f.max_jump
            return reject("spike",
                f"jump={f.max_jump:.4f} >= th={th_spike:.4f} "
                f"(over by {-margin:.4f})", margin)

        # Gate 6: Bayesian
        p_win, _ = self.bayes.predict(f.fv, f.regime)
        if p_win < th_bayes:
            margin = p_win - th_bayes
            return reject("bayes",
                f"p_win={p_win:.4f} < th={th_bayes:.4f} "
                f"(short by {-margin:.4f})", margin)

        # All passed
        stake, score = self._stake(p_win, balance, f, th_vol, th_range, th_z, risk)
        marti_info   = (f" [MARTI step={risk.marti_step}]" if risk and risk.marti_step > 0
                        else "")
        log.info(
            f"SIGNAL | tick={f.tick} p={p_win:.3f} "
            f"sig*sqrtT={f.sigma_sqrt_T:.3f}/{th_vol:.3f} "
            f"rng={f.range_width:.3f}/{th_range:.3f} "
            f"|Z|={abs(f.zscore):.3f}/{th_z:.3f} "
            f"stake={stake:.2f}{marti_info}"
        )
        return Decision(trade=True, stake=stake, p_win=p_win, score=score,
                        reason="all gates passed", gate="", margin=0.0,
                        thresholds=thresholds)

    def _stake(self, p: float, balance: float,
               f: Features, th_vol: float, th_range: float, th_z: float,
               risk: Optional["RiskManager"] = None) -> Tuple[float, float]:
        b     = self.cfg.payout_ratio
        q     = 1 - p
        kelly = max(0.0, (p * b - q) / b)
        frac  = min(kelly * self.cfg.kelly_frac, 0.05)
        raw   = frac * balance
        kelly_stake = max(self.cfg.min_stake, min(raw, self.cfg.max_stake))

        if risk and risk.marti_step > 0:
            stake = min(risk.martingale_stake, self.cfg.max_stake)
        else:
            stake = kelly_stake

        vol_score   = max(0, 1 - f.sigma_sqrt_T / th_vol) if th_vol else 0
        range_score = max(0, 1 - f.range_width  / th_range) if th_range else 0
        z_score_val = max(0, 1 - abs(f.zscore)  / th_z) if th_z else 0
        score       = round((vol_score + range_score + z_score_val) / 3, 3)

        return round(stake, 2), score


# -----------------------------------------------------------------------------
# RISK MANAGER
# -----------------------------------------------------------------------------

class RiskManager:

    def __init__(self, cfg: Config):
        self.cfg               = cfg
        self._consec_losses    = 0
        self._last_loss_time:  Optional[float] = None
        self._in_trade         = False
        self._paused           = False
        self._pause_reason     = ""
        self._start_balance:   Optional[float] = None
        self._daily_pnl        = 0.0
        self._marti_step       = 0

    def set_balance(self, b: float):
        if self._start_balance is None:
            self._start_balance = b

    @property
    def in_trade(self) -> bool:
        return self._in_trade

    def can_trade(self, balance: float = 0.0) -> Tuple[bool, str]:
        if self._in_trade:
            return False, "in_trade"
        if self._paused:
            return False, f"paused:{self._pause_reason}"
        if self._last_loss_time:
            waited = time.time() - self._last_loss_time
            if waited < self.cfg.cooldown_after_loss:
                remaining = int(self.cfg.cooldown_after_loss - waited)
                return False, f"cooldown:{remaining}s"
        if self._consec_losses >= self.cfg.max_consecutive_losses:
            self._paused       = True
            self._pause_reason = f"{self._consec_losses}_consec_losses"
            return False, f"paused:{self._pause_reason}"
        if self._start_balance:
            cap = self._start_balance * self.cfg.max_daily_loss_pct
            if self._daily_pnl < -cap:
                self._paused       = True
                self._pause_reason = "daily_loss_cap"
                return False, "paused:daily_loss_cap"
        min_safe = self.cfg.min_stake * self.cfg.balance_guard_multiples
        if balance > 0 and balance < min_safe:
            return False, f"balance_too_low:{balance:.2f}<{min_safe:.2f}"
        return True, "ok"

    def on_open(self):
        self._in_trade = True

    def on_close(self, won: bool, profit: float):
        self._in_trade   = False
        self._daily_pnl += profit
        if won:
            if self._marti_step > 0:
                log.info(f"MARTINGALE RESET | win at step={self._marti_step} → back to base")
            self._consec_losses  = 0
            self._last_loss_time = None
            self._marti_step     = 0
        else:
            self._consec_losses  += 1
            self._last_loss_time  = time.time()
            if self._consec_losses >= self.cfg.marti_kick_in:
                if self._marti_step < self.cfg.marti_max_steps:
                    self._marti_step += 1
                    next_stake = self.cfg.min_stake * (self.cfg.marti_factor ** self._marti_step)
                    log.info(
                        f"MARTINGALE STEP {self._marti_step}/{self.cfg.marti_max_steps}"
                        f" | loss #{self._consec_losses}"
                        f" | next_stake≈${next_stake:.2f}"
                    )
                else:
                    log.warning(
                        f"MARTINGALE MAX STEP reached ({self._marti_step}) — "
                        f"stake stays at "
                        f"${self.cfg.min_stake * (self.cfg.marti_factor ** self._marti_step):.2f}"
                        f" until win or hard reset"
                    )
            else:
                log.info(
                    f"LOSS #{self._consec_losses} | martingale inactive "
                    f"(kicks in after {self.cfg.marti_kick_in} losses)"
                )

    @property
    def martingale_stake(self) -> float:
        return round(self.cfg.min_stake * (self.cfg.marti_factor ** self._marti_step), 2)

    @property
    def marti_step(self) -> int:
        return self._marti_step

    def release_trade_lock(self):
        """
        Release the in-trade lock without recording a win or loss.
        Used when settlement cannot be confirmed — martingale state unchanged.
        """
        self._in_trade = False
        log.warning("Trade lock released (unconfirmed outcome — martingale unchanged)")

    def reset(self):
        self._paused         = False
        self._consec_losses  = 0
        self._last_loss_time = None
        self._marti_step     = 0
        log.info("RiskManager: hard reset — martingale and loss streak cleared")


# -----------------------------------------------------------------------------
# TRADE HISTORY
# -----------------------------------------------------------------------------

class History:

    COLS = ["ts", "contract_id", "stake", "p_win", "score", "regime",
            "sigma_sqrtT", "range_width", "zscore",
            "vol_th", "range_th", "bayes_th",
            "won", "profit", "balance", "settle_source"]

    def __init__(self, path: str):
        self.path  = path
        self._rows: List[dict] = []
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.COLS).writeheader()

    def add(self, row: dict):
        self._rows.append(row)
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.COLS).writerow(
                {c: row.get(c, "") for c in self.COLS}
            )

    def update_last(self, contract_id, won: bool, profit: float,
                    balance: float, settle_source: str = "api"):
        for r in reversed(self._rows):
            if str(r.get("contract_id")) == str(contract_id):
                r["won"]           = won
                r["profit"]        = round(profit, 5)
                r["balance"]       = round(balance, 4)
                r["settle_source"] = settle_source
                self._rewrite()
                return

    def _rewrite(self):
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.COLS)
            w.writeheader()
            for r in self._rows:
                w.writerow({c: r.get(c, "") for c in self.COLS})

    @property
    def stats(self) -> dict:
        done = [r for r in self._rows if r.get("won") != ""]
        if not done:
            return {"n": 0, "win_rate": 0.0, "pnl": 0.0}
        wins = sum(1 for r in done if r.get("won") is True or r.get("won") == "True")
        pnl  = sum(float(r.get("profit", 0) or 0) for r in done)
        return {"n": len(done), "win_rate": wins / len(done), "pnl": round(pnl, 4)}


# -----------------------------------------------------------------------------
# DERIV WEBSOCKET CLIENT
# -----------------------------------------------------------------------------

class DerivClient:

    def __init__(self, cfg: Config):
        self.cfg      = cfg
        self._ws      = None
        self._rid     = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._tick_cb: Optional[Callable]         = None
        self._connected: bool = False
        self.balance: float = 0.0

    async def connect(self):
        url             = f"{self.cfg.api_url}?app_id={self.cfg.app_id}"
        self._ws        = await websockets.connect(url, ping_interval=20, ping_timeout=10)
        self._connected = True
        asyncio.create_task(self._listen())

    async def auth(self):
        r = await self._rpc({"authorize": self.cfg.api_token})
        if "error" in r:
            raise ConnectionError(r["error"]["message"])
        self.balance = float(r["authorize"].get("balance", 0))
        log.info(f"Auth OK | login={r['authorize'].get('loginid')} balance={self.balance}")

    async def subscribe_ticks(self, cb: Callable):
        self._tick_cb = cb
        await self._send({"ticks": self.cfg.symbol, "subscribe": 1, "req_id": self._next()})

    async def proposal(self, stake: float) -> Optional[dict]:
        r = await self._rpc({
            "proposal":      1,
            "amount":        str(stake),
            "basis":         "stake",
            "contract_type": self.cfg.contract_type,
            "currency":      self.cfg.currency,
            "duration":      self.cfg.expiry_min,
            "duration_unit": "m",
            "symbol":        self.cfg.symbol,
            "barrier":       f"+{self.cfg.barrier}",
            "barrier2":      f"-{self.cfg.barrier}",
        })
        if "error" in r:
            log.warning(f"Proposal error: {r['error']['message']}")
            return None
        return r.get("proposal")

    async def buy(self, proposal_id: str, price: float) -> Optional[dict]:
        r = await self._rpc({"buy": proposal_id, "price": str(price)})
        if "error" in r:
            log.error(f"Buy error: {r['error']['message']}")
            return None
        b            = r.get("buy", {})
        self.balance = float(b.get("balance_after", self.balance))
        return b

    async def contract_status(self, contract_id) -> Optional[dict]:
        r = await self._rpc({
            "proposal_open_contract": 1,
            "contract_id": int(contract_id),
        })
        if "error" in r:
            log.debug(f"contract_status error for {contract_id}: {r['error'].get('message')}")
            return None
        return r.get("proposal_open_contract")

    async def profit_table_lookup(self, contract_id) -> Optional[dict]:
        r = await self._rpc({
            "profit_table": 1,
            "description":  1,
            "sort":         "DESC",
            "limit":        10,
        })
        for txn in r.get("profit_table", {}).get("transactions", []):
            if str(txn.get("contract_id")) == str(contract_id):
                return txn
        return None

    async def refresh_balance(self):
        r            = await self._rpc({"balance": 1, "account": "current"})
        self.balance = float(r.get("balance", {}).get("balance", self.balance))

    async def disconnect(self):
        if self._ws:
            await self._ws.close()

    @property
    def connected(self) -> bool:
        return self._connected

    def _next(self) -> int:
        self._rid += 1
        return self._rid

    async def _rpc(self, payload: dict) -> dict:
        # FIX: use get_running_loop() — correct on Python 3.10+
        loop             = asyncio.get_running_loop()
        rid              = self._next()
        payload["req_id"] = rid
        fut              = loop.create_future()
        self._pending[rid] = fut
        await self._send(payload)
        try:
            return await asyncio.wait_for(fut, timeout=20.0)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            return {"error": {"message": "timeout"}}

    async def _send(self, payload: dict):
        await self._ws.send(json.dumps(payload))

    async def _listen(self):
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                if msg.get("msg_type") == "tick" and self._tick_cb:
                    q = float(msg.get("tick", {}).get("quote", 0))
                    if q > 0:
                        asyncio.create_task(self._call(q))
                    continue
                rid = msg.get("req_id")
                if rid and rid in self._pending:
                    f = self._pending.pop(rid)
                    if not f.done():
                        f.set_result(msg)
        except Exception as e:
            log.error(f"WebSocket listener error: {e}")
        finally:
            self._connected = False
            log.warning("WebSocket listener exited — connection lost")

    async def _call(self, price: float):
        try:
            if asyncio.iscoroutinefunction(self._tick_cb):
                await self._tick_cb(price)
            else:
                self._tick_cb(price)
        except Exception as e:
            log.error(f"Tick cb error: {e}")


# -----------------------------------------------------------------------------
# BOT ENGINE
# -----------------------------------------------------------------------------

class Bot:

    def __init__(self, cfg: Config):
        self.cfg     = cfg
        self.buf     = TickBuffer(cfg)
        self.bayes   = BayesModel(cfg)
        self.chain   = GateChain(cfg, self.buf, self.bayes)
        self.risk    = RiskManager(cfg)
        self.history = History(cfg.history_file)
        self.client  = DerivClient(cfg)
        self._alive  = True

        # Skip logging
        self._last_skip_log:      float   = 0.0
        self._skip_counts:        Counter = Counter()
        self._ticks_after_warmup: int     = 0

        # State log
        self._last_feature_log: float = 0.0

        # Deadlock detector
        # FIX: drought counter only increments when NOT in a trade.
        # Previously it incremented every tick including during the 2-min
        # contract window, causing false deadlock triggers.
        self._ticks_since_trade:        int = 0
        self._deadlock_relief_remaining: int = 0

    # ── Entry ──────────────────────────────────────────────────────────────────

    async def run(self):
        self.bayes.load(self.cfg.state_file)
        await self._connect_and_run()

    async def _connect_and_run(self):
        retry_delay = 5
        while self._alive:
            try:
                log.info("Connecting to Deriv API...")
                await self.client.connect()
                await self.client.auth()
                self.risk.set_balance(self.client.balance)
                log.info(self.bayes.summary())
                if not self.buf.is_warm:
                    log.info(f"Warming up — need {self.cfg.warmup_ticks} ticks before trading")
                else:
                    log.info(f"Buffer already warm ({self.buf.tick} ticks) — trading immediately")
                await self.client.subscribe_ticks(self.on_tick)
                retry_delay = 5
                while self._alive and self.client.connected:
                    await asyncio.sleep(1)
                if self._alive:
                    log.warning("Connection lost — reconnecting in 5s...")
                    await asyncio.sleep(5)
            except Exception as e:
                log.error(f"Connection error: {e} — retrying in {retry_delay}s")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
        # FIX: removed duplicate _save() here — shutdown() already saves before
        # setting _alive=False, so this path is only reached after a clean shutdown.
        await self.client.disconnect()

    # ── Tick handler ───────────────────────────────────────────────────────────

    async def on_tick(self, price: float):
        f = self.buf.push(price)
        if f is None:
            remaining = self.cfg.warmup_ticks - self.buf.tick
            if self.buf.tick % 50 == 0:
                log.info(f"Warming up: {remaining} ticks remaining...")
            return

        self._ticks_after_warmup += 1

        # Periodic state log (every 15 s)
        now = time.time()
        if now - self._last_feature_log > 15:
            self._log_state(f)
            self._last_feature_log = now

        # Periodic skip summary — independent of skip rate-limiter
        if self._ticks_after_warmup % self.cfg.skip_summary_every == 0:
            self._log_skip_summary()

        # ── Single authoritative gate check ───────────────────────────────────
        # FIX: removed the direct risk._in_trade check that preceded can_trade().
        # can_trade() is the single gate; accessing _in_trade directly was
        # bypassing the public API and creating a confusing double-check.
        ok, reason = self.risk.can_trade(self.client.balance)
        if not ok:
            self._skip_counts[reason] += 1
            # FIX: only increment drought counter when bot is genuinely idle
            # (not blocked because a trade is already running).
            if reason != "in_trade":
                self._check_deadlock()
            self._maybe_log_skip_reason(reason, f)
            return

        # Bot is idle and all risk checks pass — run deadlock check and gate chain
        self._check_deadlock()

        dec = self.chain.evaluate(f, self.client.balance, self.risk,
                                  widen_factor=self._deadlock_widen_factor)
        if not dec.trade:
            self._skip_counts[f"gate:{dec.gate}"] += 1
            self._maybe_log_skip(dec, f)
            return

        # FIX: _execute is dispatched as a Task so on_tick returns immediately.
        # The 2-minute settlement sleep happens in the background; tick processing
        # continues uninterrupted.  risk.on_open() is called inside _execute
        # before the task yields, so subsequent ticks see in_trade=True instantly.
        asyncio.create_task(self._execute(dec, f))

    # ── Skip logging ───────────────────────────────────────────────────────────

    def _maybe_log_skip(self, dec: Decision, f: Features):
        now = time.time()
        if now - self._last_skip_log < self.cfg.skip_log_interval:
            return
        self._last_skip_log = now
        th = dec.thresholds
        log.info(
            f"[SKIP tick={f.tick}] gate={dec.gate} | {dec.reason} | "
            f"regime={f.regime} | "
            f"sig*sqrtT={f.sigma_sqrt_T:.4f}/{th.get('vol','?')} "
            f"rng={f.range_width:.4f}/{th.get('range','?')} "
            f"ema={f.ema_distance:.4f}/{th.get('ema','?')} "
            f"|Z|={abs(f.zscore):.4f}/{th.get('z','?')} "
            f"jump={f.max_jump:.4f}/{th.get('spike','?')} "
            f"bayes_th={th.get('bayes','?')}"
        )

    def _maybe_log_skip_reason(self, reason: str, f: Features):
        now = time.time()
        if now - self._last_skip_log < self.cfg.skip_log_interval:
            return
        self._last_skip_log = now
        log.info(f"[SKIP tick={f.tick}] reason={reason}")

    def _log_skip_summary(self):
        total = sum(self._skip_counts.values())
        if total == 0:
            return
        summary = " | ".join(
            f"{k}:{v}({v/total*100:.0f}%)"
            for k, v in self._skip_counts.most_common()
        )
        log.info(
            f"[SKIP SUMMARY ticks={self._ticks_after_warmup}] "
            f"total_skipped={total} | {summary}"
        )

    # ── Trade execution ────────────────────────────────────────────────────────

    async def _execute(self, dec: Decision, f: Features):
        """
        FIX: Settlement sleep moved out of on_tick coroutine.
        This method runs as an independent asyncio Task, so the 2-minute
        wait does not block tick processing or gate evaluation.
        risk.on_open() is called synchronously at the top so that subsequent
        ticks immediately see in_trade=True.
        """
        prop = await self.client.proposal(dec.stake)
        if not prop:
            return

        # Set in-trade lock before any await so the very next tick sees it
        self.risk.on_open()
        self._ticks_since_trade = 0   # reset deadlock drought counter on trade entry

        result = await self.client.buy(prop["id"], float(prop["ask_price"]))
        if not result:
            # Buy failed — release lock and treat as loss for risk accounting
            self.risk.on_close(won=False, profit=-dec.stake)
            return

        cid       = result.get("contract_id")
        buy_price = float(result.get("buy_price", dec.stake))

        self.history.add({
            "ts":          datetime.utcnow().isoformat(),
            "contract_id": cid,
            "stake":       buy_price,
            "p_win":       round(dec.p_win, 4),
            "score":       dec.score,
            "regime":      f.regime,
            "sigma_sqrtT": round(f.sigma_sqrt_T, 4),
            "range_width": round(f.range_width, 4),
            "zscore":      round(f.zscore, 4),
            "vol_th":      dec.thresholds.get("vol"),
            "range_th":    dec.thresholds.get("range"),
            "bayes_th":    dec.thresholds.get("bayes"),
        })

        # Wait for nominal expiry + buffer — runs in background Task
        wait = self.cfg.expiry_min * 60 + self.cfg.settle_wait_extra
        log.info(f"Contract open | id={cid} stake={buy_price:.2f} | waiting {wait}s...")
        await asyncio.sleep(wait)

        await self._settle(cid, buy_price, f)

    # ── Settlement with verification ───────────────────────────────────────────

    async def _settle(self, cid, buy_price: float, f: Features):
        """
        Polls the API until the contract is confirmed settled.

        FIX: wrapped in try/finally so the trade lock is always released even
        if an exception occurs during polling (e.g. WS drop mid-settle).
        Previously an exception here would leave _in_trade=True permanently,
        deadlocking the bot until restart.

        Flow:
          1. Poll proposal_open_contract every SETTLE_POLL_INTERVAL seconds.
          2. If that fails, check profit_table as fallback.
          3. Record outcome + update Bayes only on API confirmation.
          4. If neither path confirms, log WARNING and skip Bayes update.
        """
        won           = None
        profit        = None
        settle_source = "unknown"

        log.info(f"[SETTLE] Polling contract {cid} for settlement confirmation...")

        try:
            for attempt in range(1, self.cfg.settle_poll_attempts + 1):
                status = await self.client.contract_status(cid)

                if status:
                    is_sold    = status.get("is_sold", False)
                    sell_price = status.get("sell_price")
                    api_profit = status.get("profit")
                    api_status = status.get("status", "")

                    log.info(
                        f"[SETTLE] Poll {attempt}/{self.cfg.settle_poll_attempts} | "
                        f"cid={cid} status={api_status!r} "
                        f"is_sold={is_sold} sell_price={sell_price} profit={api_profit}"
                    )

                    if is_sold or api_status in ("sold", "won", "lost"):
                        if api_profit is not None:
                            profit = float(api_profit)
                        elif sell_price is not None:
                            profit = float(sell_price) - buy_price
                        else:
                            profit = 0.0

                        if profit == 0.0 and sell_price is not None:
                            profit = float(sell_price) - buy_price

                        won           = profit > 0
                        settle_source = "proposal_open_contract"
                        break
                else:
                    log.info(
                        f"[SETTLE] Poll {attempt}/{self.cfg.settle_poll_attempts} | "
                        f"cid={cid} — no response, retrying in {self.cfg.settle_poll_interval}s"
                    )

                await asyncio.sleep(self.cfg.settle_poll_interval)

            # Fallback: profit_table
            if won is None:
                log.warning(
                    f"[SETTLE] proposal_open_contract did not confirm for {cid} "
                    f"after {self.cfg.settle_poll_attempts} polls — trying profit_table"
                )
                balance_before = self.client.balance
                txn = await self.client.profit_table_lookup(cid)
                if txn:
                    profit        = float(txn.get("profit", 0))
                    settle_source = "profit_table"

                    if profit == 0.0:
                        await self.client.refresh_balance()
                        delta = round(self.client.balance - balance_before, 5)
                        if delta > 0:
                            profit = delta
                            won    = True
                            log.info(
                                f"[SETTLE] profit_table profit=0 RESOLVED via balance delta "
                                f"(Δ={delta:+.5f}) → WON | cid={cid}"
                            )
                        elif delta < 0:
                            profit = delta
                            won    = False
                            log.info(
                                f"[SETTLE] profit_table profit=0 RESOLVED via balance delta "
                                f"(Δ={delta:+.5f}) → LOST | cid={cid}"
                            )
                        else:
                            log.warning(
                                f"[SETTLE] profit_table profit=0 AND balance delta=0 for "
                                f"{cid} — outcome ambiguous (possible void/refund). "
                                f"Skipping Bayes update."
                            )
                            # Release lock via finally; won=None skips Bayes below
                            return
                    else:
                        won = profit > 0
                        await self.client.refresh_balance()

                    log.info(
                        f"[SETTLE] profit_table confirmed | cid={cid} "
                        f"profit={profit:+.4f} won={won}"
                    )
                else:
                    log.warning(
                        f"[SETTLE] ⚠ UNCONFIRMED — contract {cid} not found in "
                        f"profit_table either. Skipping Bayes update."
                    )
                    # Release lock via finally; won=None skips Bayes below
                    return

            # Confirmed settlement
            if settle_source != "profit_table":
                await self.client.refresh_balance()

            self.risk.on_close(won, profit)
            self.history.update_last(cid, won, profit, self.client.balance, settle_source)
            self.bayes.update(f.fv, f.regime, won)
            self._save()

            stats = self.history.stats
            log.info(
                f"{'WIN' if won else 'LOSS'} | cid={cid} profit={profit:+.4f} "
                f"balance={self.client.balance:.2f} source={settle_source} | "
                f"win_rate={stats['win_rate']:.1%} n={stats['n']}"
            )
            log.info(self.bayes.summary())

        except Exception as exc:
            log.error(f"[SETTLE] Exception during settlement for {cid}: {exc}", exc_info=True)
            # Fall through to finally — lock will be released
        finally:
            # FIX: if settlement was not confirmed (won is still None), or an
            # exception was raised, ensure the trade lock is always released so
            # the bot can continue trading on the next tick.
            if won is None:
                self.risk.release_trade_lock()

    # ── Deadlock detector ──────────────────────────────────────────────────────

    @property
    def _deadlock_widen_factor(self) -> float:
        if self._deadlock_relief_remaining <= 0:
            return 1.0
        cfg      = self.cfg
        progress = 1.0 - (self._deadlock_relief_remaining / cfg.deadlock_relief_ticks)
        return cfg.deadlock_widen_factor - progress * (cfg.deadlock_widen_factor - 1.0)

    def _check_deadlock(self):
        """
        FIX: only called when the bot is genuinely idle (not in a trade).
        Previously called on every tick including during active contracts,
        causing the drought counter to inflate during normal 2-min holds.
        """
        cfg = self.cfg
        self._ticks_since_trade += 1

        if self._deadlock_relief_remaining > 0:
            self._deadlock_relief_remaining -= 1
            if self._deadlock_relief_remaining == 0:
                log.info("[DEADLOCK] Relief window ended — thresholds back to normal auto-cal")
            return

        if self._ticks_since_trade >= cfg.deadlock_scan_limit:
            log.warning(
                f"[DEADLOCK] {self._ticks_since_trade} ticks without a trade — "
                f"widening all thresholds by {cfg.deadlock_widen_factor}× "
                f"for {cfg.deadlock_relief_ticks} ticks."
            )
            self._deadlock_relief_remaining = cfg.deadlock_relief_ticks
            self._ticks_since_trade = 0

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _log_state(self, f: Features):
        wf = self._deadlock_widen_factor
        th = {
            "vol":   round(self.buf.vol_threshold()             * wf, 4),
            "range": round(self.buf.range_threshold()           * wf, 4),
            "ema":   round(self.buf.ema_threshold()             * wf, 4),
            "spike": round(self.buf.spike_threshold()           * wf, 4),
            "z":     round(self.buf.zscore_limit(f.sigma_price) * wf, 4),
        }
        cfg    = self.cfg
        z_hist = list(self.buf._hist_zscore_abs)
        z_dist = round(
            percentile(z_hist, cfg.zscore_floor_percentile)
            if len(z_hist) >= 30 else cfg.zscore_floor_max, 4
        )
        z_cov  = round(
            cfg.z_coverage_factor * (cfg.barrier / f.sigma_price)
            if f.sigma_price > 0 else cfg.zscore_floor_max, 4
        )
        deadlock_info = (
            f" [DEADLOCK RELIEF: {self._deadlock_relief_remaining}tk widen={wf:.2f}x]"
            if self._deadlock_relief_remaining > 0 else
            f" [drought={self._ticks_since_trade}/{cfg.deadlock_scan_limit}]"
        )
        log.info(
            f"[STATE tick={f.tick}] price={f.price:.4f} "
            f"sig*sqrtT={f.sigma_sqrt_T:.4f} "
            f"rng={f.range_width:.4f} |Z|={abs(f.zscore):.4f} regime={f.regime}"
            f"{deadlock_info}\n"
            f"  thresholds(auto{'+widen' if wf > 1.0 else ''}): {th}\n"
            f"  z_limit detail: dist_p{cfg.zscore_floor_percentile:.0f}={z_dist} "
            f"cov_cap={z_cov} history_n={len(z_hist)} "
            f"-> effective={th['z']}"
        )

    def _save(self):
        self.bayes.save(self.cfg.state_file)

    def shutdown(self):
        self._alive = False
        self._save()
        self._log_skip_summary()
        log.info(f"Shutdown | stats={self.history.stats}")


# -----------------------------------------------------------------------------
# BACKTESTER
# -----------------------------------------------------------------------------

def run_backtest(cfg: Config, n_ticks: int = 8000, seed: int = 42):
    random.seed(seed)
    print("=" * 64)
    print("Backtest: 1HZ10V ExpiryRange +/-1.9 | 2-min | self-calibrating")
    print("=" * 64)

    def gen_ticks(n, base=9800.0):
        prices = [base]
        sigma  = 0.04
        for i in range(n - 1):
            if random.random() < 0.004:
                sigma = random.choice([0.02, 0.035, 0.05, 0.08, 0.13, 0.20])
            drift = random.gauss(0, sigma) + (base - prices[-1]) * 0.002
            prices.append(round(prices[-1] + drift, 5))
        return prices

    ticks  = gen_ticks(n_ticks)
    buf    = TickBuffer(cfg)
    bayes  = BayesModel(cfg)
    chain  = GateChain(cfg, buf, bayes)

    balance             = 1000.0
    balance_log         = [balance]
    trades              = 0
    wins                = 0
    skip_until          = 0
    regime_tally: Dict[str, List[bool]] = {"low": [], "medium": [], "high": []}
    pred_vs_act:  List[Tuple[float, bool]] = []
    skip_counts:  Counter = Counter()

    for i, price in enumerate(ticks):
        f = buf.push(price)
        if f is None or i < skip_until:
            continue

        dec = chain.evaluate(f, balance, widen_factor=1.0)
        if not dec.trade:
            skip_counts[dec.gate] += 1
            continue

        future = ticks[i: min(i + cfg.t_ticks, len(ticks))]
        won    = all(abs(p - price) <= cfg.barrier for p in future)
        profit = dec.stake * cfg.payout_ratio if won else -dec.stake

        balance += profit
        balance_log.append(balance)
        bayes.update(f.fv, f.regime, won)

        trades += 1
        if won:
            wins += 1
        regime_tally[f.regime].append(won)
        pred_vs_act.append((dec.p_win, won))
        skip_until = i + 25

        if trades % 50 == 0:
            print(f"  [tick {i:5d}] trades={trades} "
                  f"win={wins/trades:.1%} bal={balance:.2f}")

    wr  = wins / trades if trades else 0.0
    pnl = balance - 1000.0
    peaks = [max(balance_log[:i+1]) for i in range(len(balance_log))]
    dd    = max((p - v) / p for p, v in zip(peaks, balance_log)) if len(balance_log) > 1 else 0.0

    print(f"\n{'-'*64}")
    print(f"  Trades      : {trades}")
    print(f"  Win rate    : {wr:.1%}")
    print(f"  P&L         : {pnl:+.2f} USD  (start 1000)")
    print(f"  Max drawdown: {dd:.1%}")

    if skip_counts:
        total = sum(skip_counts.values())
        print(f"\n  Skip breakdown ({total} total skips):")
        for gate, count in skip_counts.most_common():
            print(f"    {gate:10s}: {count} ({count/total:.0%})")

    print(f"\n  Win rate by regime:")
    for r, results in regime_tally.items():
        if results:
            print(f"    {r:8s}: {sum(results)/len(results):.1%} ({sum(results)}/{len(results)})")

    print(f"\n  Auto-calibrated thresholds at end of run:")
    print(f"    vol_th  : {buf.vol_threshold():.4f}")
    print(f"    range_th: {buf.range_threshold():.4f}")
    print(f"    ema_th  : {buf.ema_threshold():.4f}")
    print(f"    spike_th: {buf.spike_threshold():.4f}")
    z_hist = list(buf._hist_zscore_abs)
    z_dist = percentile(z_hist, cfg.zscore_floor_percentile) if len(z_hist) >= 30 else cfg.zscore_floor_max
    print(f"    z_floor : dist_p{cfg.zscore_floor_percentile:.0f}={z_dist:.4f} "
          f"history_n={len(z_hist)}")

    print(f"\n  Final Bayesian model:\n{bayes.summary()}")
    print("=" * 64)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def _start_health_server():
    import http.server, threading
    port = int(os.getenv("PORT", "8080"))

    class _H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK - bot running")
        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("", port), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log.info(f"Health-check server listening on :{port}")


async def live(cfg: Config):
    if not cfg.api_token:
        log.error("Set DERIV_API_TOKEN environment variable first.")
        sys.exit(1)

    bot = Bot(cfg)

    def handle_signal(sig, frame):
        log.info("Shutdown signal received...")
        bot.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT,  handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    log.info("=" * 64)
    log.info(f"Deriv ExpiryRange Bot — {cfg.symbol} +/-{cfg.barrier} {cfg.expiry_min}min")
    log.info("Thresholds: AUTO-CALIBRATED from live tick distribution")
    log.info("Settlement: background Task — ticks continue during contract window")
    log.info("Deadlock detector: ENABLED (idle-only counter)")
    log.info("=" * 64)

    _start_health_server()
    await bot.run()


if __name__ == "__main__":
    cfg = Config()

    if "--backtest" in sys.argv:
        run_backtest(cfg)
    else:
        asyncio.run(live(cfg))
