from datamodel import OrderDepth, TradingState, Order
from typing import List

class Trader:
    
    def __init__(self):
        # EMA state — one value per product, persists across ticks
        self.ema = {}
        self.alpha = 0.3  # EMA smoothing factor — we'll discuss tuning later
        
        # Known fair values for stable products
        # Emeralds = 10000, confirmed from your tutorial data
        self.fair_values = {
            "EMERALDS": 10000
        }
        
        # Position limits — check your wiki for exact numbers
        # These are typical values from past editions
        self.position_limits = {
            "EMERALDS": 50,
            "TOMATOES": 50
        }
    
    def run(self, state: TradingState):
        result = {}
        
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            if product == "EMERALDS":
                orders = self.trade_stable(
                    product, state, order_depth
                )
            
            elif product == "TOMATOES":
                orders = self.trade_noisy(
                    product, state, order_depth
                )
            
            result[product] = orders
        
        # Return: orders dict, conversions (0 for now), log string
        return result, 0, ""
    
    # --------------------------------------------------------
    # STRATEGY 1: Stable product (Emeralds)
    # Fair value is known = 10000
    # Quote just inside, manage inventory carefully
    # --------------------------------------------------------
    def trade_stable(self, product, state, order_depth):
        orders = []
        fair_value = self.fair_values[product]
        position = state.position.get(product, 0)
        limit = self.position_limits[product]
        
        # How much room do we have to buy and sell?
        buy_capacity  = limit - position   # max units we can still buy
        sell_capacity = limit + position   # max units we can still sell
        
        # --- TAKE profitable orders from the book first ---
        # If someone is selling below fair value, buy from them immediately
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_vol = order_depth.sell_orders[best_ask]
            
            if best_ask < fair_value and buy_capacity > 0:
                # Take as much as we can
                qty = min(buy_capacity, -best_ask_vol)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
        
        # If someone is buying above fair value, sell to them immediately
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_vol = order_depth.buy_orders[best_bid]
            
            if best_bid > fair_value and sell_capacity > 0:
                qty = min(sell_capacity, best_bid_vol)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
        
        # --- MAKE markets around fair value ---
        # Post a bid just below fair value
        # Post an ask just above fair value
        # Skew based on position to manage inventory
        
        # Inventory skew: if we're long, make ask more aggressive
        # If we're short, make bid more aggressive
        skew = -position // 10  # gentle skew
        
        bid_price = fair_value - 1 + skew
        ask_price = fair_value + 1 + skew
        
        # Only post if we have capacity
        if buy_capacity > 0:
            orders.append(Order(product, bid_price, buy_capacity // 2))
        
        if sell_capacity > 0:
            orders.append(Order(product, ask_price, -(sell_capacity // 2)))
        
        return orders
    
    # --------------------------------------------------------
    # STRATEGY 2: Noisy product (Tomatoes)
    # Fair value estimated using EMA of mid price
    # Same MM logic but with dynamic fair value
    # --------------------------------------------------------
    def trade_noisy(self, product, state, order_depth):
        orders = []
        position = state.position.get(product, 0)
        limit = self.position_limits.get(product, 50)
        
        # Calculate current mid price from order book
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders  # no market, skip this tick
        
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        
        # Update EMA with new mid price
        if product not in self.ema:
            self.ema[product] = mid_price
        else:
            self.ema[product] = (
                self.alpha * mid_price +
                (1 - self.alpha) * self.ema[product]
            )
        
        fair_value = self.ema[product]
        
        buy_capacity  = limit - position
        sell_capacity = limit + position
        
        # Take mispriced orders
        if order_depth.sell_orders:
            best_ask_price = min(order_depth.sell_orders.keys())
            best_ask_vol   = order_depth.sell_orders[best_ask_price]
            if best_ask_price < fair_value - 1 and buy_capacity > 0:
                qty = min(buy_capacity, -best_ask_vol)
                if qty > 0:
                    orders.append(Order(product, best_ask_price, qty))
        
        if order_depth.buy_orders:
            best_bid_price = max(order_depth.buy_orders.keys())
            best_bid_vol   = order_depth.buy_orders[best_bid_price]
            if best_bid_price > fair_value + 1 and sell_capacity > 0:
                qty = min(sell_capacity, best_bid_vol)
                if qty > 0:
                    orders.append(Order(product, best_bid_price, -qty))
        
        # Make markets with wider spread (more uncertainty)
        skew = -position // 10
        bid_price = round(fair_value) - 2 + skew
        ask_price = round(fair_value) + 2 + skew
        
        if buy_capacity > 0:
            orders.append(Order(product, bid_price, buy_capacity // 2))
        if sell_capacity > 0:
            orders.append(Order(product, ask_price, -(sell_capacity // 2)))
        
        return orders