import pandas as pd
import numpy as np

class SimpleBacktester:
    def __init__(self, initial_cash=1000000, commission=0.0006, slippage_ticks=3):
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage_ticks = slippage_ticks

    def run(self, pred_df, data_df, topk=50):
        """
        pred_df: index=[datetime, instrument], columns=[score]
        data_df: index=[datetime, instrument], columns=[open, high, low, close, volume, tick_size]
        """
        df = pred_df.copy()
        # Ensure tick_size is in the joined data
        df = df.join(data_df[['close', 'tick_size']], how='inner')
        df = df.sort_index()

        datetimes = df.index.get_level_values('datetime').unique()

        cash = self.initial_cash
        positions = {} # instrument -> amount
        history = []

        for dt in datetimes:
            current_data = df.xs(dt, level='datetime')
            top_k_instruments = current_data['score'].nlargest(topk).index.tolist()

            # Calculate current portfolio value
            portfolio_value = cash
            for inst, amount in positions.items():
                if inst in current_data.index:
                    price = current_data.loc[inst, 'close']
                else:
                    # Fallback if instrument data missing for this timestamp
                    # (Should not happen with inner join, but for safety)
                    price = data_df.xs(dt, level='datetime').loc[inst, 'close']
                portfolio_value += amount * price

            history.append({
                'datetime': dt,
                'portfolio_value': portfolio_value,
                'cash': cash
            })

            target_value_per_inst = portfolio_value / topk if topk > 0 else 0

            # Sell
            to_sell = [inst for inst in positions if inst not in top_k_instruments]
            for inst in to_sell:
                price = current_data.loc[inst, 'close']
                tick_size = current_data.loc[inst, 'tick_size']
                sell_price = price - self.slippage_ticks * tick_size
                amount = positions[inst]
                cash += amount * sell_price * (1 - self.commission)
                del positions[inst]

            # Rebalance / Buy
            for inst in top_k_instruments:
                price = current_data.loc[inst, 'close']
                tick_size = current_data.loc[inst, 'tick_size']

                current_amount = positions.get(inst, 0)
                # For buying, use higher price due to slippage
                buy_price = price + self.slippage_ticks * tick_size
                # For selling/rebalancing down, use lower price
                sell_price = price - self.slippage_ticks * tick_size

                target_amount = target_value_per_inst / buy_price if buy_price > 0 else 0
                diff = target_amount - current_amount

                if diff > 0:
                    cost = diff * buy_price * (1 + self.commission)
                    if cash >= cost:
                        cash -= cost
                        positions[inst] = target_amount
                elif diff < 0:
                    sell_amount = -diff
                    cash += sell_amount * sell_price * (1 - self.commission)
                    positions[inst] = target_amount

        return pd.DataFrame(history).set_index('datetime')
