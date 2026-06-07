import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class Analyzer:
    @staticmethod
    def calculate_ic(pred_df, label_df):
        combined = pred_df.join(label_df, how='inner')
        ic = combined.groupby(level='datetime').apply(lambda x: x['score'].corr(x['label']))
        rank_ic = combined.groupby(level='datetime').apply(lambda x: x['score'].corr(x['label'], method='spearman'))
        return ic, rank_ic

    @staticmethod
    def get_backtest_metrics(report_df):
        if report_df.empty:
            return {'Sharpe Ratio': 0, 'Profit Factor': 0, 'Cumulative Return': 0}

        returns = report_df['portfolio_value'].pct_change().dropna()

        cumulative_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns).mean()**(365 * 24 * 60) - 1

        cum_max = report_df['portfolio_value'].cummax()
        drawdown = (report_df['portfolio_value'] - cum_max) / cum_max
        max_drawdown = drawdown.min()

        volatility = returns.std() * np.sqrt(365 * 24 * 60)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Profit Factor: Sum(Positive Returns) / Abs(Sum(Negative Returns))
        pos_returns = returns[returns > 0].sum()
        neg_returns = abs(returns[returns < 0].sum())
        profit_factor = pos_returns / neg_returns if neg_returns > 0 else (np.inf if pos_returns > 0 else 1.0)

        return {
            'Cumulative Return': float(cumulative_return),
            'Annualized Return': float(annualized_return),
            'Max Drawdown': float(max_drawdown),
            'Sharpe Ratio': float(sharpe_ratio),
            'Volatility': float(volatility),
            'Profit Factor': float(profit_factor)
        }

    @staticmethod
    def generate_report(report_df, ic, rank_ic, output_path='report.html'):
        fig = make_subplots(rows=3, cols=1,
                           subplot_titles=("Cumulative Return", "Drawdown", "IC Distribution"),
                           vertical_spacing=0.1)
        fig.add_trace(go.Scatter(x=report_df.index, y=report_df['portfolio_value'], name="Portfolio Value"), row=1, col=1)
        cum_max = report_df['portfolio_value'].cummax()
        drawdown = (report_df['portfolio_value'] - cum_max) / cum_max
        fig.add_trace(go.Scatter(x=report_df.index, y=drawdown, name="Drawdown", fill='tozeroy'), row=2, col=1)
        fig.add_trace(go.Histogram(x=ic, name="IC", nbinsx=50), row=3, col=1)
        fig.update_layout(height=1000, title_text="Crypto Qlib Backtest Report", showlegend=True)
        fig.write_html(output_path)
        print(f"Report saved to {output_path}")
