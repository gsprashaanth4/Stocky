class Portfolio:

    def __init__(
        self,
        initial_cash=10_000.0,
        fee_rate=0.001,
        slippage_rate=0.001
    ):

        if initial_cash <= 0:
            raise ValueError("Initial cash must be positive.")

        if not 0 <= fee_rate < 1:
            raise ValueError("Invalid fee rate.")

        if not 0 <= slippage_rate < 1:
            raise ValueError("Invalid slippage rate.")

        self.initial_cash = float(initial_cash)

        self.cash = float(initial_cash)
        self.shares = 0.0

        # Accrued but not yet paid dividends.
        self.dividend_receivable = 0.0

        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate

        self.total_fees = 0.0
        self.total_dividends = 0.0

        self.trade_history = []

    # -----------------------------------------------
    # Portfolio valuation
    # -----------------------------------------------

    def position_value(self, price):

        return self.shares * price

    def equity(self, price):

        return (
            self.cash
            + self.position_value(price)
            + self.dividend_receivable
        )

    def allocation(self, price):

        total_equity = self.equity(price)

        if total_equity <= 0:
            return 0.0

        return self.position_value(price) / total_equity

    def cash_ratio(self, price):

        total_equity = self.equity(price)

        if total_equity <= 0:
            return 0.0

        return self.cash / total_equity

    # -----------------------------------------------
    # Corporate actions
    # -----------------------------------------------

    def apply_dividend(self, dividend_per_share):

        """
        Call on the ex-dividend date BEFORE trading.

        The shares held before ex-date determine entitlement.
        """

        if dividend_per_share < 0:
            raise ValueError("Dividend cannot be negative.")

        dividend = self.shares * dividend_per_share

        self.dividend_receivable += dividend
        self.total_dividends += dividend

        return dividend

    def apply_split(self, split_ratio):

        """
        Example:
            2-for-1 split -> split_ratio = 2.0
            4-for-1 split -> split_ratio = 4.0
            1-for-10 reverse split -> split_ratio = 0.1
        """

        if split_ratio <= 0:
            raise ValueError("Invalid split ratio.")

        if split_ratio == 1:
            return

        self.shares *= split_ratio

    # -----------------------------------------------
    # Rebalance portfolio
    # -----------------------------------------------

    def rebalance(self, target_allocation, market_price):

        if not 0.0 <= target_allocation <= 1.0:
            raise ValueError("Target allocation must be in [0, 1].")
    
        if market_price <= 0:
            raise ValueError("Market price must be positive.")
    
        equity_before = self.equity(market_price)
    
        if equity_before <= 0:
            raise RuntimeError("Portfolio has no positive equity.")
    
        current_position = self.shares * market_price
        desired_position = target_allocation * equity_before
    
        delta = desired_position - current_position
    
        # -------------------------------------------
        # BUY
        # -------------------------------------------
    
        if delta > 1e-12:
        
            # Total proportional cost relative to market price.
            cost_factor = (
                (1 + self.slippage_rate) *
                (1 + self.fee_rate) - 1
            )
    
            # Solve for quantity such that the post-trade
            # allocation equals target_allocation.
            quantity = (
                desired_position - current_position
            ) / (
                market_price *
                (1 + target_allocation * cost_factor)
            )
    
            fill_price = market_price * (1 + self.slippage_rate)
    
            max_affordable = self.cash / (
                fill_price * (1 + self.fee_rate)
            )
    
            quantity = min(quantity, max_affordable)
    
            trade_value = quantity * fill_price
            fee = trade_value * self.fee_rate
    
            self.cash -= trade_value + fee
            self.shares += quantity
    
            trade_type = "BUY"
    
        # -------------------------------------------
        # SELL
        # -------------------------------------------
    
        elif delta < -1e-12:
        
            cost_factor = (
                1 - (1 - self.slippage_rate) *
                (1 - self.fee_rate)
            )
    
            quantity = (
                current_position - desired_position
            ) / (
                market_price *
                (1 - target_allocation * cost_factor)
            )
    
            quantity = min(quantity, self.shares)
    
            fill_price = market_price * (1 - self.slippage_rate)
    
            trade_value = quantity * fill_price
            fee = trade_value * self.fee_rate
    
            self.cash += trade_value - fee
            self.shares -= quantity
    
            trade_type = "SELL"
    
        else:
        
            quantity = 0.0
            fill_price = market_price
            trade_value = 0.0
            fee = 0.0
    
            trade_type = "HOLD"
    
        self.total_fees += fee
    
        if abs(self.shares) < 1e-12:
            self.shares = 0.0
    
        if abs(self.cash) < 1e-8:
            self.cash = 0.0
    
        if self.cash < -1e-8 or self.shares < -1e-8:
            raise RuntimeError("Invalid portfolio accounting.")
    
        self.trade_history.append({
            "type": trade_type,
            "quantity": quantity,
            "market_price": market_price,
            "fill_price": fill_price,
            "trade_value": trade_value,
            "fee": fee,
            "cash_after": self.cash,
            "shares_after": self.shares
        })
    
        return {
            "type": trade_type,
            "quantity": quantity,
            "fill_price": fill_price,
            "fee": fee,
            "equity_after": self.equity(market_price),
            "allocation_after": self.allocation(market_price)
        }