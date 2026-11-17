# import time
# import numpy as np
# from req.WhatevPlot import WhatevPlot as wp
# import pandas as pd
# import yfinance as yf



# class Envir:
#     def __init__(self, buffer_size=252):
#         self.buffer_size = buffer_size

#         self.plots = wp(
#             title="Stocky", 
#             titles=["Just1", "Just2", "Just3", "Just4"], 
#             buffer_size=self.buffer_size, 
#             break_after=3, 
#             fps=10,
#             winx=1200, 
#             winy=800
#         )

#         self.prev = 0
#         self.alpha = 0.99

#     def step(self, action):
#         time.sleep(0.01)
#         v = self.prev*self.alpha + np.random.rand()*(1-self.alpha)
#         self.prev = v
#         self.plots.update([v,v,v,v])

#     def portfolio(self):


#     def close(self):
#         self.plots.close()

#     def reward(self):

#         self.reward = np.log(portfolio_after/portfolio_before)

#         if (buy):
#             if strong_bullish:
#                 self.reward += 0.001
#             elif strong_bearish:
#                 self.reward -= 0.001

#         elif (hold):
#             if strong_bullish:
#                 self.reward += 0.001
#             elif strong_bearish:
#                 self.reward -= 0.001

#         elif (sell):
#             if (potential_exhaustion):
#                 self.reward += 0.001
#             if (strong_bearish):
#                 self.reward += 0.001
#             if (downside_risk):
#                 self.reward += 0.001
#             if (strong_bullish):
#                 self.reward -= 0.001





import numpy as np
import gymnasium as gym

from gymnasium import spaces

from req.portfolio import Portfolio
from req.features import FEATURE_COLUMNS


class StockEnvironment(gym.Env):

    metadata = {"render_modes": []}

    ACTIONS = np.array(
        [0.0, 0.25, 0.50, 0.75, 1.0],
        dtype=np.float32
    )

    def __init__(
        self,
        ticker,
        data,
        normalized_features,
        lookback=20,
        episode_length=252,
        initial_cash=10_000.0,
        fee_rate=0.001,
        slippage_rate=0.001
    ):

        super().__init__()

        self.ticker = ticker

        self.data = data.reset_index(drop=True).copy()
        self.features = normalized_features.reset_index(drop=True).copy()

        self.lookback = lookback
        self.episode_length = episode_length

        self.initial_cash = initial_cash
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate

        # -------------------------------------------
        # Validate input data
        # -------------------------------------------

        if len(self.data) != len(self.features):
            raise ValueError(
                "Raw data and normalized features must have equal lengths."
            )

        if not all(
            col in self.features.columns
            for col in FEATURE_COLUMNS
        ):
            raise ValueError("Missing normalized feature columns.")

        if len(self.data) < lookback + episode_length:
            raise ValueError(
                f"{ticker}: insufficient data for the requested "
                f"lookback ({lookback}) and episode length "
                f"({episode_length})."
            )

        if self.data["Date"].duplicated().any():
            raise ValueError(f"{ticker}: duplicate dates found.")

        if not self.data["Date"].is_monotonic_increasing:
            raise ValueError(f"{ticker}: data is not chronological.")

        if not self.data["Date"].equals(self.features["Date"]):
            # Normalized features normally don't contain Date.
            # Date alignment is checked by row order instead.
            pass

        # -------------------------------------------
        # Action and observation spaces
        # -------------------------------------------

        self.action_space = spaces.Discrete(len(self.ACTIONS))

        observation_size = (
            lookback * len(FEATURE_COLUMNS) + 2
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32
        )

        # -------------------------------------------
        # Random episode starting point
        # -------------------------------------------

        self.np_random = np.random.default_rng()

        self.min_start = lookback - 1

        self.max_start = (
            len(self.data) - 1 - episode_length
        )

        # Runtime variables.
        self.portfolio = None
        self.current_idx = None
        self.steps = 0

    # ===============================================
    # Helpers
    # ===============================================

    def _get_price(self, idx, column):

        return float(self.data.iloc[idx][column])

    def _get_observation(self):

        start = self.current_idx - self.lookback + 1
        end = self.current_idx + 1

        # 20 rows x 7 features.
        market_state = self.features.iloc[
            start:end
        ][FEATURE_COLUMNS].to_numpy(dtype=np.float32)

        if market_state.shape != (
            self.lookback,
            len(FEATURE_COLUMNS)
        ):
            raise RuntimeError("Invalid market observation shape.")

        market_state = market_state.flatten()

        close = self._get_price(
            self.current_idx,
            "Close"
        )

        portfolio_state = np.array([
            self.portfolio.allocation(close),
            self.portfolio.cash_ratio(close)
        ], dtype=np.float32)

        state = np.concatenate([
            market_state,
            portfolio_state
        ])

        return state.astype(np.float32)

    # ===============================================
    # Reset
    # ===============================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        options = options or {}

        # Allow deterministic starting points for testing.
        if "start_idx" in options:

            start_idx = int(options["start_idx"])

            if not (
                self.min_start <= start_idx <= self.max_start
            ):
                raise ValueError("Invalid episode starting index.")

            self.current_idx = start_idx

        else:

            self.current_idx = int(
                self.np_random.integers(
                    self.min_start,
                    self.max_start + 1
                )
            )

        # Every episode gets a fresh portfolio.
        self.portfolio = Portfolio(
            initial_cash=self.initial_cash,
            fee_rate=self.fee_rate,
            slippage_rate=self.slippage_rate
        )

        self.steps = 0

        observation = self._get_observation()

        info = {
            "ticker": self.ticker,
            "date": self.data.iloc[self.current_idx]["Date"],
            "equity": self.initial_cash,
            "allocation": 0.0
        }

        return observation, info

    # ===============================================
    # Step
    # ===============================================

    def step(self, action):

        if self.portfolio is None:
            raise RuntimeError("Call reset() before step().")

        if not self.action_space.contains(action):
            # raise ValueError(f"Invalid action: {action}")

            if not isinstance(action, (int, float, np.integer, np.floating)):
                raise ValueError(f"Invalid action type: {type(action)}")
            
            action = float(action)
            
            if not np.isfinite(action) or not 0.0 <= action <= 1.0:
                raise ValueError(f"Invalid allocation: {action}")

        # -------------------------------------------
        # Current portfolio value at previous close
        # -------------------------------------------

        previous_close = self._get_price(
            self.current_idx,
            "Close"
        )

        equity_before = self.portfolio.equity(
            previous_close
        )

        # -------------------------------------------
        # Advance one trading session
        # -------------------------------------------

        next_idx = self.current_idx + 1

        next_row = self.data.iloc[next_idx]

        # -------------------------------------------
        # Corporate actions
        # -------------------------------------------

        # Apply split before dividend and trading.
        # This adjusts the share count to the new
        # share basis before processing the ex-date.

        split_ratio = float(next_row["Stock Splits"])

        if split_ratio > 0:
            self.portfolio.apply_split(split_ratio)

        # Dividend entitlement is based on holdings
        # before today's opening trade.

        dividend_per_share = float(
            next_row["Dividends"]
        )

        if dividend_per_share > 0:

            self.portfolio.apply_dividend(
                dividend_per_share
            )

        # -------------------------------------------
        # Execute action at next day's OPEN
        # -------------------------------------------

        target_allocation = float(
            self.ACTIONS[int(action)]
        )

        next_open = float(next_row["Open"])

        trade_info = self.portfolio.rebalance(
            target_allocation=target_allocation,
            market_price=next_open
        )

        # -------------------------------------------
        # Mark portfolio at next day's CLOSE
        # -------------------------------------------

        next_close = float(next_row["Close"])

        equity_after = self.portfolio.equity(
            next_close
        )

        # -------------------------------------------
        # Reward
        # -------------------------------------------

        if equity_before <= 0 or equity_after <= 0:
            raise RuntimeError(
                "Non-positive portfolio equity encountered."
            )

        reward = float(
            np.log(equity_after / equity_before)
        )

        # -------------------------------------------
        # Update environment
        # -------------------------------------------

        self.current_idx = next_idx
        self.steps += 1

        observation = self._get_observation()

        # Dataset/episode boundary is a truncation,
        # not an economic termination.
        terminated = False

        truncated = (
            self.steps >= self.episode_length
            or self.current_idx >= len(self.data) - 1
        )

        info = {
            "ticker": self.ticker,
            "date": next_row["Date"],

            "action": int(action),
            "target_allocation": target_allocation,

            "equity_before": equity_before,
            "equity_after": equity_after,

            "cash": self.portfolio.cash,
            "shares": self.portfolio.shares,

            "allocation": self.portfolio.allocation(next_close),

            "dividend_receivable": self.portfolio.dividend_receivable,

            "dividend": dividend_per_share,
            "split": split_ratio,

            "trade": trade_info
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )

    # ===============================================
    # Close
    # ===============================================

    def close(self):
        pass