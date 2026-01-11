"""
Backtesting and trading simulator.
Implements paper trading with $10,000 starting capital, realistic slippage and fees.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import json

class TradingSimulator:
    """
    Paper trading simulator with realistic execution.
    
    Execution price: P_exec = P_t * (1 + slippage_bps * sign(order))
    where slippage_bps is basis points (e.g., 0.0005 for 5 bps)
    """
    
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 slippage_bps: float = 0.0005,  # 5 basis points
                 commission_per_trade: float = 1.0,
                 max_position_pct: float = 0.20):  # Max 20% per position
        
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.slippage_bps = slippage_bps
        self.commission = commission_per_trade
        self.max_position_pct = max_position_pct
        
        self.positions = {}  # ticker -> quantity
        self.trades = []
        self.portfolio_values = []
        
    def get_portfolio_value(self, date: str, prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value.
        Portfolio value = cash + sum(positions * current_price)
        """
        position_value = sum(
            qty * prices.get(ticker, 0) 
            for ticker, qty in self.positions.items()
        )
        return self.cash + position_value
    
    def execute_trade(self, 
                     ticker: str, 
                     quantity: int, 
                     price: float, 
                     date: str,
                     reason: str = "signal") -> bool:
        """
        Execute a trade with slippage and commission.
        
        Args:
            ticker: Stock ticker
            quantity: Number of shares (positive for buy, negative for sell)
            price: Current market price
            date: Trade date
            reason: Reason for trade
            
        Returns:
            True if trade executed, False if insufficient funds
        """
        if quantity == 0:
            return False
        
        # Apply slippage
        # Slippage increases cost for buys, decreases proceeds for sells
        exec_price = price * (1 + self.slippage_bps * np.sign(quantity))
        
        # Calculate total cost including commission
        trade_value = quantity * exec_price
        total_cost = abs(trade_value) + self.commission
        
        # Check if we have enough cash for buys
        if quantity > 0 and total_cost > self.cash:
            return False
        
        # Execute trade
        self.cash -= trade_value + self.commission
        
        if ticker not in self.positions:
            self.positions[ticker] = 0
        self.positions[ticker] += quantity
        
        # Remove position if closed
        if self.positions[ticker] == 0:
            del self.positions[ticker]
        
        # Log trade
        trade_log = {
            'date': date,
            'ticker': ticker,
            'action': 'BUY' if quantity > 0 else 'SELL',
            'quantity': abs(quantity),
            'price': price,
            'exec_price': exec_price,
            'slippage': abs(exec_price - price),
            'commission': self.commission,
            'total_cost': total_cost,
            'cash_after': self.cash,
            'reason': reason
        }
        self.trades.append(trade_log)
        
        return True
    
    def calculate_position_size(self, 
                                ticker: str, 
                                signal_strength: float, 
                                current_price: float,
                                portfolio_value: float) -> int:
        """
        Calculate position size based on signal strength and risk limits.
        
        Uses fixed fraction of portfolio value, capped at max_position_pct.
        """
        # Allocate based on signal strength (0 to 1)
        allocation = min(abs(signal_strength) * self.max_position_pct, self.max_position_pct)
        
        # Dollar amount to allocate
        dollar_allocation = portfolio_value * allocation
        
        # Number of shares (floor to integer)
        shares = int(dollar_allocation / current_price)
        
        return shares if signal_strength > 0 else -shares
    
    def run_backtest(self, 
                    predictions_df: pd.DataFrame, 
                    prices_df: pd.DataFrame,
                    strategy: str = 'top_k',
                    top_k: int = 5) -> Dict:
        """
        Run backtest using predictions.
        
        Args:
            predictions_df: DataFrame with columns [date, ticker, pred_1d, lower, upper]
            prices_df: DataFrame with actual prices [date, ticker, adj_close]
            strategy: Trading strategy ('top_k', 'threshold')
            top_k: For top_k strategy, number of stocks to hold
            
        Returns:
            Backtest results and performance metrics
        """
        print(f"[v0] Running backtest with strategy={strategy}, top_k={top_k}...")
        
        # Get unique dates sorted
        dates = sorted(predictions_df['date'].unique())
        
        for date in dates:
            # Get predictions for this date
            day_preds = predictions_df[predictions_df['date'] == date]
            
            # Get current prices
            day_prices = prices_df[prices_df['date'] == date]
            price_dict = dict(zip(day_prices['ticker'], day_prices['adj_close']))
            
            # Calculate portfolio value
            portfolio_value = self.get_portfolio_value(date, price_dict)
            self.portfolio_values.append({
                'date': date,
                'value': portfolio_value,
                'cash': self.cash,
                'positions': len(self.positions)
            })
            
            # Top-K strategy: buy top K predicted performers
            if strategy == 'top_k':
                # Sort by predicted return
                day_preds = day_preds.sort_values('pred_1d', ascending=False)
                
                # Top K to buy
                top_tickers = set(day_preds.head(top_k)['ticker'].values)
                
                # Sell positions not in top K
                for ticker in list(self.positions.keys()):
                    if ticker not in top_tickers and ticker in price_dict:
                        qty = self.positions[ticker]
                        self.execute_trade(ticker, -qty, price_dict[ticker], date, reason='rebalance_sell')
                
                # Buy or increase positions in top K
                for _, row in day_preds.head(top_k).iterrows():
                    ticker = row['ticker']
                    pred = row['pred_1d']
                    
                    if ticker not in price_dict:
                        continue
                    
                    current_price = price_dict[ticker]
                    current_qty = self.positions.get(ticker, 0)
                    
                    # Calculate target position
                    signal_strength = min(max(pred * 10, 0), 1)  # Scale prediction to 0-1
                    target_qty = self.calculate_position_size(ticker, signal_strength, current_price, portfolio_value)
                    
                    # Trade to reach target
                    trade_qty = target_qty - current_qty
                    if trade_qty != 0:
                        self.execute_trade(ticker, trade_qty, current_price, date, reason='rebalance_buy')
        
        # Final portfolio value
        final_date = dates[-1]
        final_prices = prices_df[prices_df['date'] == final_date]
        final_price_dict = dict(zip(final_prices['ticker'], final_prices['adj_close']))
        final_value = self.get_portfolio_value(final_date, final_price_dict)
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics()
        
        print(f"\n[v0] Backtest complete!")
        print(f"[v0] Initial capital: ${self.initial_capital:,.2f}")
        print(f"[v0] Final value: ${final_value:,.2f}")
        print(f"[v0] Total return: {(final_value / self.initial_capital - 1) * 100:.2f}%")
        print(f"[v0] Total trades: {len(self.trades)}")
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': (final_value / self.initial_capital - 1),
            'trades': self.trades,
            'portfolio_values': self.portfolio_values,
            'metrics': metrics
        }
    
    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate financial performance metrics.
        
        Metrics:
        - CAGR = (V_end / V_start)^(1/T) - 1
        - Sharpe = mean_annual_return / std_annual_return
        - Max Drawdown = max peak-to-trough decline
        """
        if len(self.portfolio_values) < 2:
            return {}
        
        values = [pv['value'] for pv in self.portfolio_values]
        dates = [pv['date'] for pv in self.portfolio_values]
        
        # Daily returns
        returns = np.diff(values) / values[:-1]
        
        # CAGR
        days = (pd.to_datetime(dates[-1]) - pd.to_datetime(dates[0])).days
        years = days / 365.25
        cagr = (values[-1] / values[0]) ** (1 / years) - 1 if years > 0 else 0
        
        # Annualized return and volatility
        mean_daily_return = np.mean(returns)
        std_daily_return = np.std(returns)
        annual_return = mean_daily_return * 252
        annual_volatility = std_daily_return * np.sqrt(252)
        
        # Sharpe ratio (assuming 0 risk-free rate)
        sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Max drawdown
        cummax = np.maximum.accumulate(values)
        drawdowns = (values - cummax) / cummax
        max_drawdown = np.min(drawdowns)
        
        return {
            'cagr': cagr,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades)
        }


def main():
    """Run backtesting simulation."""
    # Load features and train a simple model for predictions
    df = pd.read_parquet('data/features.parquet')
    df = df.dropna(subset=['target_1d'])
    
    print(f"[v0] Loaded {len(df)} rows")
    
    # For demo, use a simple model
    from models import LightGBMPredictor
    
    # Train model on first 80% of data
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"[v0] Training on {len(train_df)} rows, testing on {len(test_df)} rows")
    
    model = LightGBMPredictor(target='target_1d')
    X_train, y_train = model.prepare_features(train_df)
    model.train(X_train, y_train)
    
    # Generate predictions for test period
    X_test, y_test = model.prepare_features(test_df)
    preds, lower, upper = model.predict_with_uncertainty(X_test)
    
    # Create predictions DataFrame
    predictions_df = test_df[['date', 'ticker']].copy()
    predictions_df['pred_1d'] = preds
    predictions_df['lower'] = lower
    predictions_df['upper'] = upper
    
    # Prices DataFrame
    prices_df = test_df[['date', 'ticker', 'adj_close']].copy()
    
    # Run backtest
    simulator = TradingSimulator(initial_capital=10000.0)
    results = simulator.run_backtest(predictions_df, prices_df, strategy='top_k', top_k=5)
    
    # Save results
    with open('data/backtest_results.json', 'w') as f:
        # Convert to JSON-serializable format
        results_json = results.copy()
        results_json['trades'] = results['trades'][:100]  # Limit for file size
        json.dump(results_json, f, indent=2, default=str)
    
    print(f"\n[v0] Results saved to data/backtest_results.json")
    print(f"[v0] CAGR: {results['metrics'].get('cagr', 0)*100:.2f}%")
    print(f"[v0] Sharpe Ratio: {results['metrics'].get('sharpe_ratio', 0):.2f}")
    print(f"[v0] Max Drawdown: {results['metrics'].get('max_drawdown', 0)*100:.2f}%")

if __name__ == '__main__':
    main()
