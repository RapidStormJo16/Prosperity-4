"""
IMC Prosperity 4 - Round 3 Algorithm
Products: HYDROGEL_PACK, VELVETFRUIT_EXTRACT, VEV_* (10 option strikes)

Strategy:
  - HYDROGEL_PACK: Market making around rolling fair value (mean ~9991, sigma ~32)
  - VELVETFRUIT_EXTRACT: Market making around rolling fair value (mean ~5250)
  - VEV options: Black-Scholes based market making using VEX as underlying
    * Implied vol is stable at ~23.5% across all historical days
    * TTE decreases: Round 3 = TTE 5 days (in-sim), decreasing each timestep
    * Skip VEV_6000 and VEV_6500 (effectively worthless, price stuck at 0.5)
"""

from datamodel import (
    OrderDepth, TradingState, Order, ConversionObservation,
    Observation, ProsperityEncoder, Symbol, Product, Position
)
from typing import Dict, List, Any, Tuple, Optional
import math
import json


# ─── Black-Scholes helpers ────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-Scholes call price. Returns intrinsic value if T <= 0."""
    if T <= 1e-9 or sigma <= 1e-9:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-Scholes delta (dC/dS). Used for delta-hedging VEVs via VEX."""
    if T <= 1e-9 or sigma <= 1e-9:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)


# ─── Constants ────────────────────────────────────────────────────────────────

# Position limits per product
POSITION_LIMITS = {
    "HYDROGEL_PACK":        200,
    "VELVETFRUIT_EXTRACT":  200,
    "VEV_4000":  300,
    "VEV_4500":  300,
    "VEV_5000":  300,
    "VEV_5100":  300,
    "VEV_5200":  300,
    "VEV_5300":  300,
    "VEV_5400":  300,
    "VEV_5500":  300,
    "VEV_6000":  300,
    "VEV_6500":  300,
}

# Strike prices for each VEV symbol
VEV_STRIKES = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
    "VEV_6000": 6000,
    "VEV_6500": 6500,
}

# VEVs we actively trade (skip far OTM worthless ones)
# VEV_4000 and VEV_4500: deep ITM, nearly delta-1, small spread relative to price → trade carefully
# VEV_5000 to VEV_5500: main opportunities
# VEV_6000 and VEV_6500: essentially 0 value, skip
ACTIVE_VEVS = {
    "VEV_4000", "VEV_4500",
    "VEV_5000", "VEV_5100", "VEV_5200",
    "VEV_5300", "VEV_5400", "VEV_5500",
}

# Calibrated implied volatility from historical data (extremely stable at ~23.6%)
IV = 0.236

# Risk-free rate (competition uses 0)
RISK_FREE = 0.0

# Total simulation timesteps per day in the competition (1,000,000 timestamps per day)
# TTE starts at 5 days for Round 3 and decreases to 0 over the round
# Round 3 = day 3 in the sim. Each timestamp = 1/1000000 of a day (for TTE calculation)
# Historical data: day0=TTE8, day1=TTE7, day2=TTE6 → Round3 live = TTE5 at start
TTE_ROUND3_START_DAYS = 5
TOTAL_TIMESTAMPS = 1_000_000  # timestamps per simulated day

# Market-making spreads (ticks on each side of fair value)
MM_SPREAD_HP  = 5    # Hydrogel: bid = fair-5, ask = fair+5 (spread=10, market spread~16)
MM_SPREAD_VEX = 2    # VEX: bid = fair-2, ask = fair+2 (spread=4, market spread~5)

# VEV quoting spread: half-spread in absolute price units, varies by strike
VEV_SPREAD = {
    "VEV_4000": 5,
    "VEV_4500": 5,
    "VEV_5000": 4,
    "VEV_5100": 3,
    "VEV_5200": 3,
    "VEV_5300": 2,
    "VEV_5400": 1,
    "VEV_5500": 1,
}

# Quote sizes (volume per order)
QUOTE_SIZE_HP  = 10
QUOTE_SIZE_VEX = 8
QUOTE_SIZE_VEV = 5   # smaller for options (wider P&L impact per unit)

# Inventory skew parameters: if position is too far from 0, shift our quotes to unwind
SKEW_FACTOR_HP  = 0.05  # per unit of net position, shift fair value by this many ticks
SKEW_FACTOR_VEX = 0.03
SKEW_FACTOR_VEV = 0.5

# ─── Trader class ─────────────────────────────────────────────────────────────

class Trader:

    def __init__(self):
        # Rolling mid-price tracker for HP and VEX
        self._mid_prices: Dict[str, List[float]] = {}
        self._window = 20  # how many ticks to smooth over

    # ── Utility: get best bid/ask from order book ─────────────────────────────

    @staticmethod
    def _best_bid(order_depth: OrderDepth) -> Optional[float]:
        if order_depth.buy_orders:
            return max(order_depth.buy_orders.keys())
        return None

    @staticmethod
    def _best_ask(order_depth: OrderDepth) -> Optional[float]:
        if order_depth.sell_orders:
            return min(order_depth.sell_orders.keys())
        return None

    @staticmethod
    def _mid_price(order_depth: OrderDepth) -> Optional[float]:
        bb = Trader._best_bid(order_depth)
        ba = Trader._best_ask(order_depth)
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return None

    # ── Utility: safe position lookup ─────────────────────────────────────────

    @staticmethod
    def _pos(state: TradingState, symbol: str) -> int:
        return state.position.get(symbol, 0)

    # ── Utility: safe capacity (how many more we can buy/sell) ────────────────

    @staticmethod
    def _capacity(state: TradingState, symbol: str) -> Tuple[int, int]:
        """Returns (buy_capacity, sell_capacity) given current position."""
        pos = Trader._pos(state, symbol)
        limit = POSITION_LIMITS[symbol]
        return limit - pos, limit + pos  # can buy (limit-pos) more, sell (limit+pos) more

    # ── Utility: update rolling mid price ─────────────────────────────────────

    def _update_mid(self, symbol: str, mid: float):
        if symbol not in self._mid_prices:
            self._mid_prices[symbol] = []
        self._mid_prices[symbol].append(mid)
        if len(self._mid_prices[symbol]) > self._window:
            self._mid_prices[symbol].pop(0)

    def _smooth_mid(self, symbol: str, fallback: float) -> float:
        prices = self._mid_prices.get(symbol, [])
        if not prices:
            return fallback
        return sum(prices) / len(prices)

    # ── Time to expiry calculation ─────────────────────────────────────────────

    @staticmethod
    def _tte_years(timestamp: int) -> float:
        """
        In the live simulation, timestamp goes 0 → 1,000,000 within one round.
        TTE at start of Round 3 = 5 days.
        Each timestamp is (1/1,000,000) of a day.
        So TTE in days = 5 - timestamp/1,000,000
        We convert to years: /365
        """
        tte_days = TTE_ROUND3_START_DAYS - timestamp / TOTAL_TIMESTAMPS
        tte_days = max(tte_days, 0.0)
        return tte_days / 365.0

    # ── Market making for delta-1 products (HP and VEX) ───────────────────────

    def _mm_orders(
        self,
        state: TradingState,
        symbol: str,
        spread: int,
        quote_size: int,
        skew_factor: float,
    ) -> List[Order]:
        orders: List[Order] = []
        od = state.order_depths.get(symbol)
        if od is None:
            return orders

        mid = self._mid_price(od)
        if mid is None:
            return orders

        self._update_mid(symbol, mid)
        fair = self._smooth_mid(symbol, mid)

        # Inventory skew: if long, lower our bid/ask to encourage selling
        pos = self._pos(state, symbol)
        skew = -pos * skew_factor
        fair_skewed = fair + skew

        bid_price = round(fair_skewed - spread)
        ask_price = round(fair_skewed + spread)

        buy_cap, sell_cap = self._capacity(state, symbol)

        # Buy orders
        buy_qty = min(quote_size, buy_cap)
        if buy_qty > 0 and bid_price > 0:
            # Also try to take liquidity if ask is below our fair value
            best_ask = self._best_ask(od)
            if best_ask is not None and best_ask < fair_skewed - 1:
                take_qty = min(
                    buy_cap,
                    abs(od.sell_orders.get(best_ask, 0))
                )
                if take_qty > 0:
                    orders.append(Order(symbol, best_ask, take_qty))
                    buy_cap -= take_qty
                    buy_qty = min(quote_size, buy_cap)

            if buy_qty > 0 and bid_price > 0:
                orders.append(Order(symbol, bid_price, buy_qty))

        # Sell orders
        sell_qty = min(quote_size, sell_cap)
        if sell_qty > 0:
            # Take liquidity if bid is above fair
            best_bid = self._best_bid(od)
            if best_bid is not None and best_bid > fair_skewed + 1:
                take_qty = min(
                    sell_cap,
                    od.buy_orders.get(best_bid, 0)
                )
                if take_qty > 0:
                    orders.append(Order(symbol, best_bid, -take_qty))
                    sell_cap -= take_qty
                    sell_qty = min(quote_size, sell_cap)

            if sell_qty > 0 and ask_price > 0:
                orders.append(Order(symbol, ask_price, -sell_qty))

        return orders

    # ── VEV option orders (Black-Scholes based) ────────────────────────────────

    def _vev_orders(
        self,
        state: TradingState,
        symbol: str,
        vex_mid: float,
    ) -> List[Order]:
        orders: List[Order] = []
        if symbol not in ACTIVE_VEVS:
            return orders

        od = state.order_depths.get(symbol)
        if od is None:
            return orders

        K = VEV_STRIKES[symbol]
        T = self._tte_years(state.timestamp)

        # Theoretical fair value via Black-Scholes
        theo = bs_call_price(vex_mid, K, T, IV, RISK_FREE)

        # If theoretical price is effectively 0 (deep OTM with tiny TTE), skip
        if theo < 0.5:
            return orders

        # Inventory skew for the option position
        pos = self._pos(state, symbol)
        skew = -pos * SKEW_FACTOR_VEV
        fair = theo + skew

        half_spread = VEV_SPREAD.get(symbol, 2)
        bid_price = math.floor(fair - half_spread)
        ask_price = math.ceil(fair + half_spread)

        # Clamp to positive prices
        bid_price = max(bid_price, 1)
        ask_price = max(ask_price, bid_price + 1)

        buy_cap, sell_cap = self._capacity(state, symbol)

        # Aggress: if market ask < our theoretical fair (option is cheap), buy it
        best_ask = self._best_ask(od)
        if best_ask is not None and best_ask < theo - half_spread:
            take_qty = min(
                QUOTE_SIZE_VEV,
                buy_cap,
                abs(od.sell_orders.get(best_ask, 0))
            )
            if take_qty > 0:
                orders.append(Order(symbol, best_ask, take_qty))
                buy_cap -= take_qty

        # Aggress: if market bid > our theoretical fair (option is expensive), sell it
        best_bid = self._best_bid(od)
        if best_bid is not None and best_bid > theo + half_spread:
            take_qty = min(
                QUOTE_SIZE_VEV,
                sell_cap,
                od.buy_orders.get(best_bid, 0)
            )
            if take_qty > 0:
                orders.append(Order(symbol, best_bid, -take_qty))
                sell_cap -= take_qty

        # Passive market-making quotes
        buy_qty = min(QUOTE_SIZE_VEV, buy_cap)
        if buy_qty > 0:
            orders.append(Order(symbol, bid_price, buy_qty))

        sell_qty = min(QUOTE_SIZE_VEV, sell_cap)
        if sell_qty > 0:
            orders.append(Order(symbol, ask_price, -sell_qty))

        return orders

    # ── Delta hedge: use VEX to hedge accumulated VEV delta ───────────────────

    def _delta_hedge_orders(
        self,
        state: TradingState,
        vex_mid: float,
    ) -> List[Order]:
        """
        Compute total delta exposure from all VEV positions and hedge via VEX.
        Net_delta = sum over all VEVs of (position_i * delta_i)
        Target VEX position = -Net_delta (rounded)
        """
        T = self._tte_years(state.timestamp)
        net_delta = 0.0

        for sym, K in VEV_STRIKES.items():
            pos = self._pos(state, sym)
            if pos == 0:
                continue
            delta = bs_delta(vex_mid, K, T, IV, RISK_FREE)
            net_delta += pos * delta

        # Target: hold -net_delta units of VEX to be delta neutral
        target_vex_pos = -round(net_delta)
        current_vex_pos = self._pos(state, "VELVETFRUIT_EXTRACT")
        needed = target_vex_pos - current_vex_pos

        if needed == 0:
            return []

        orders: List[Order] = []
        od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if od is None:
            return orders

        buy_cap, sell_cap = self._capacity(state, "VELVETFRUIT_EXTRACT")

        if needed > 0:
            # Need to buy VEX
            qty = min(needed, buy_cap, QUOTE_SIZE_VEX * 2)
            if qty > 0:
                best_ask = self._best_ask(od)
                if best_ask is not None:
                    orders.append(Order("VELVETFRUIT_EXTRACT", best_ask, qty))
        elif needed < 0:
            # Need to sell VEX
            qty = min(-needed, sell_cap, QUOTE_SIZE_VEX * 2)
            if qty > 0:
                best_bid = self._best_bid(od)
                if best_bid is not None:
                    orders.append(Order("VELVETFRUIT_EXTRACT", best_bid, -qty))

        return orders

    # ── Main run_step ──────────────────────────────────────────────────────────

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """
        Called every timestep. Returns:
          - orders: dict of symbol -> list of Order
          - conversions: int (not used this round)
          - trader_data: str (state to pass to next tick, serialized as JSON)
        """
        all_orders: Dict[str, List[Order]] = {}

        # ── 1. Get VEX fair value (needed for all VEV pricing) ────────────────
        vex_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        vex_mid = self._mid_price(vex_od) if vex_od else 5250.0
        if vex_mid is None:
            vex_mid = 5250.0  # fallback to historical mean

        self._update_mid("VELVETFRUIT_EXTRACT", vex_mid)
        vex_fair = self._smooth_mid("VELVETFRUIT_EXTRACT", vex_mid)

        # ── 2. HYDROGEL_PACK market making ────────────────────────────────────
        hp_orders = self._mm_orders(
            state, "HYDROGEL_PACK",
            spread=MM_SPREAD_HP,
            quote_size=QUOTE_SIZE_HP,
            skew_factor=SKEW_FACTOR_HP,
        )
        if hp_orders:
            all_orders["HYDROGEL_PACK"] = hp_orders

        # ── 3. VELVETFRUIT_EXTRACT market making (independent of hedge) ───────
        # We'll layer: first do MM orders, then add delta hedge on top
        vex_mm_orders = self._mm_orders(
            state, "VELVETFRUIT_EXTRACT",
            spread=MM_SPREAD_VEX,
            quote_size=QUOTE_SIZE_VEX,
            skew_factor=SKEW_FACTOR_VEX,
        )

        # ── 4. VEV option market making ───────────────────────────────────────
        for sym in VEV_STRIKES:
            vev_orders = self._vev_orders(state, sym, vex_fair)
            if vev_orders:
                all_orders[sym] = vev_orders

        # ── 5. Delta hedge via VEX ────────────────────────────────────────────
        hedge_orders = self._delta_hedge_orders(state, vex_fair)

        # Merge VEX orders: MM + hedge
        # Important: consolidate by price level to avoid duplicate prices
        combined_vex: Dict[int, int] = {}
        for o in vex_mm_orders + hedge_orders:
            price = int(o.price)
            combined_vex[price] = combined_vex.get(price, 0) + o.quantity

        vex_all_orders = [
            Order("VELVETFRUIT_EXTRACT", price, qty)
            for price, qty in combined_vex.items()
            if qty != 0
        ]
        if vex_all_orders:
            all_orders["VELVETFRUIT_EXTRACT"] = vex_all_orders

        # No conversions this round
        conversions = 0

        # Persist rolling mid prices to trader_data
        trader_data = json.dumps({"mids": self._mid_prices})

        return all_orders, conversions, trader_data
