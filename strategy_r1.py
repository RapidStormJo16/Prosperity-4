from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

class Trader:

    def __init__(self):
        # Position limits for Round 1
        self.position_limits = {
            "INTARIAN_PEPPER_ROOT": 80,
            "ASH_COATED_OSMIUM": 80
        }

        # ASH_COATED_OSMIUM: mean-reverting around 10000
        self.aco_fair_value = 10000

        # INTARIAN_PEPPER_ROOT: linear uptrend at +1 per 1000 timestamps
        # We need to track the starting price to compute the trend
        self.ipr_trend_rate = 1.0 / 1000  # price increase per timestamp unit

    def run(self, state: TradingState):
        result = {}

        # Restore persisted state from previous tick
        persisted = {}
        if state.traderData and state.traderData != "":
            try:
                persisted = json.loads(state.traderData)
            except:
                persisted = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            if product == "INTARIAN_PEPPER_ROOT":
                orders, persisted = self.trade_ipr(product, state, order_depth, persisted)
            elif product == "ASH_COATED_OSMIUM":
                orders = self.trade_aco(product, state, order_depth)

            result[product] = orders

        trader_data = json.dumps(persisted)
        return result, 0, trader_data

    # ------------------------------------------------------------------
    # INTARIAN_PEPPER_ROOT: Trend-Following Market Maker
    #
    # Price follows: fair_value ≈ start_price + timestamp / 1000
    # Strategy: always be long-biased (price only goes up),
    # aggressively buy below trend, patiently sell above trend.
    # ------------------------------------------------------------------
    def trade_ipr(self, product, state, order_depth, persisted):
        orders = []
        position = state.position.get(product, 0)
        limit = self.position_limits[product]

        buy_capacity = limit - position
        sell_capacity = limit + position

        # Estimate fair value from the linear trend
        # On the first tick, we calibrate start_price from the current order book
        if "ipr_start_price" not in persisted:
            # Bootstrap: estimate start price = current_mid - timestamp * rate
            mid = self._get_mid(order_depth)
            if mid is not None:
                persisted["ipr_start_price"] = mid - state.timestamp * self.ipr_trend_rate
            else:
                return orders, persisted  # no data yet

        start_price = persisted["ipr_start_price"]
        fair_value = start_price + state.timestamp * self.ipr_trend_rate

        # === TAKE: Sweep ALL sell orders below fair value (buy cheap) ===
        if order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price < fair_value and buy_capacity > 0:
                    ask_vol = -order_depth.sell_orders[ask_price]  # sell_orders have negative volumes
                    qty = min(buy_capacity, ask_vol)
                    if qty > 0:
                        orders.append(Order(product, ask_price, qty))
                        buy_capacity -= qty
                elif ask_price >= fair_value:
                    break  # sorted ascending, no more cheap asks

        # === TAKE: Sweep ALL buy orders above fair value (sell dear) ===
        if order_depth.buy_orders:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price > fair_value + 1 and sell_capacity > 0:
                    bid_vol = order_depth.buy_orders[bid_price]
                    qty = min(sell_capacity, bid_vol)
                    if qty > 0:
                        orders.append(Order(product, bid_price, -qty))
                        sell_capacity -= qty
                elif bid_price <= fair_value + 1:
                    break

        # === MAKE: Post quotes around fair value ===
        # Long-biased: bid aggressively (FV - 1), ask patiently (FV + 3)
        # Because price is going UP, holding inventory is profitable
        fv_int = round(fair_value)

        # Inventory skew: reduce position when getting close to limits
        skew = 0
        if position > 40:
            skew = -2  # too long, make ask more aggressive
        elif position > 20:
            skew = -1
        elif position < -40:
            skew = 2   # too short, make bid more aggressive
        elif position < -20:
            skew = 1

        bid_price = fv_int - 1 + skew
        ask_price = fv_int + 3 + skew

        # Post remaining capacity
        if buy_capacity > 0:
            orders.append(Order(product, bid_price, buy_capacity))

        if sell_capacity > 0:
            orders.append(Order(product, ask_price, -sell_capacity))

        return orders, persisted

    # ------------------------------------------------------------------
    # ASH_COATED_OSMIUM: Mean-Reversion Market Maker
    #
    # Price oscillates around 10000 with no drift.
    # Strategy: aggressively take mispriced orders at all depth levels,
    # manage inventory with strong skew back to zero.
    # ------------------------------------------------------------------
    def trade_aco(self, product, state, order_depth):
        orders = []
        position = state.position.get(product, 0)
        limit = self.position_limits[product]

        buy_capacity = limit - position
        sell_capacity = limit + position

        fair_value = self.aco_fair_value  # 10000

        # === TAKE: Sweep ALL sell orders below fair value ===
        if order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price < fair_value and buy_capacity > 0:
                    ask_vol = -order_depth.sell_orders[ask_price]
                    qty = min(buy_capacity, ask_vol)
                    if qty > 0:
                        orders.append(Order(product, ask_price, qty))
                        buy_capacity -= qty
                elif ask_price >= fair_value:
                    break

        # === TAKE: Sweep ALL buy orders above fair value ===
        if order_depth.buy_orders:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price > fair_value and sell_capacity > 0:
                    bid_vol = order_depth.buy_orders[bid_price]
                    qty = min(sell_capacity, bid_vol)
                    if qty > 0:
                        orders.append(Order(product, bid_price, -qty))
                        sell_capacity -= qty
                elif bid_price <= fair_value:
                    break

        # === MAKE: Post quotes with aggressive inventory management ===
        # Stronger skew than IPR because mean-reversion punishes holding inventory
        skew = 0
        if position > 30:
            skew = -3
        elif position > 15:
            skew = -2
        elif position > 5:
            skew = -1
        elif position < -30:
            skew = 3
        elif position < -15:
            skew = 2
        elif position < -5:
            skew = 1

        bid_price = fair_value - 2 + skew
        ask_price = fair_value + 2 + skew

        if buy_capacity > 0:
            orders.append(Order(product, bid_price, buy_capacity))

        if sell_capacity > 0:
            orders.append(Order(product, ask_price, -sell_capacity))

        return orders

    # ------------------------------------------------------------------
    # Helper: get mid price from order book
    # ------------------------------------------------------------------
    def _get_mid(self, order_depth):
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        elif order_depth.sell_orders:
            return min(order_depth.sell_orders.keys())
        elif order_depth.buy_orders:
            return max(order_depth.buy_orders.keys())
        return None
