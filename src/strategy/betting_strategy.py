from abc import ABC, abstractmethod
from typing import List
from enum import Enum
import pandas as pd
import numpy as np
from scipy.stats import norm


class BetType(Enum):
    HOME = 'H'
    DRAW = 'D'
    AWAY = 'A'

class BettingStrategy(ABC):
    def __init__(self, initial_bankroll=1000):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.history = pd.DataFrame(columns=['Match', 'Bet', 'Stake', 'Outcome', 'ProfitLoss', 'Bankroll'])
    
    @property
    def name(self):
        return self.__class__.__name__

    def _odds_to_probability(self, odds: List[float]) -> List[float]:
        """
        Convert a list of betting odds to probabilities and adjust for the overround.

        :param odds: The list of betting odds.
        :return: The list of probabilities implied by the odds, adjusted for the overround.
        """
        probabilities = [1 / odd for odd in odds]
        overround = sum(probabilities)
        probabilities_adjusted = [prob / overround for prob in probabilities]
        return probabilities_adjusted
    
    def _expected_value(self, p: float, odds: float) -> float:
        """
        Calculate the expected value of a bet.

        :param p: The estimated probability of winning.
        :param odds: The betting odds.
        :return: The expected value of the bet.
        """
        return (p * odds) - (1 - p)
    
    def get_historical_returns(self) -> pd.Series:
        return self.history['ProfitLoss'] / (self.history['Bankroll'] - self.history['ProfitLoss'])
    
    def metric_report(self):
        history = self.history

        total_profit = history['ProfitLoss'].sum()
        total_staked = history['Stake'].sum()

        # Calculating returns for each bet
        returns = self.get_historical_returns()

        # Calculating win rate
        win_rate = (history['Outcome'] == 'Win').mean()

        # Calculating Return on Investment (ROI) adn Yield
        return_on_investment = total_profit / self.initial_bankroll
        yield_rate = total_profit / total_staked 

        # Calculating Sharpe Ratio and Sortino Ratio
        # Measure of risk-adjusted return, with a higher value indicating a better risk-adjusted performance
        sharpe_ratio = returns.mean() / returns.std()
        # Measure of risk-adjusted return that penalizes only downside volatility. A higher value indicates a better risk-adjusted performance.
        sortino_ratio = returns.mean() / returns.loc[returns < returns.mean()].std()
        
        volatility = history['ProfitLoss'].std()
        std_dev_returns = returns.std()

        # Value at Risk (VaR): This is a measure of the risk of potential losses. It's typically calculated as the worst 
        # loss that you could expect to occur with a certain level of confidence (e.g., 95% or 99%) over a certain period. 
        # VaR_95 = norm.ppf(1-0.95, np.mean(returns), std_dev_returns)

        # running_total = history['ProfitLoss'].cumsum()
        # running_max = running_total.cummax()
        # drawdown = running_max - running_total
        # max_drawdown = drawdown.max()

        return {
            'Total Profit': total_profit,
            'Total Staked': total_staked,
            'Current Bankroll': self.bankroll,
            'Standard Deviation of Returns': std_dev_returns,
            'ROI': return_on_investment,
            'Yield': yield_rate,
            'Win Rate': win_rate,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
        }


    @abstractmethod
    def run(self, dataset, model, initial_bankroll: int = 1000):
        pass
