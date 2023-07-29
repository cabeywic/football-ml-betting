from abc import ABC, abstractmethod
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
        self.bankroll = initial_bankroll
        self.history = pd.DataFrame(columns=['Match', 'Bet', 'Stake', 'Outcome', 'ProfitLoss', 'Bankroll'])
    
    @property
    def name(self):
        return self.__class__.__name__

    def _odds_to_probability(self, odds: float) -> float:
        """
        Convert betting odds to probabilities.

        :param odds: The betting odds.
        :return: The probability implied by the odds.
        """
        return 1 / odds
    
    def _expected_value(self, p: float, odds: float) -> float:
        """
        Calculate the expected value of a bet.

        :param p: The estimated probability of winning.
        :param odds: The betting odds.
        :return: The expected value of the bet.
        """
        return (p * odds) - (1 - p)
    
    def metric_report(self):
        history = self.history

        total_profit = history['ProfitLoss'].sum()
        total_staked = history['Stake'].sum()

        ROI = total_profit / total_staked 
        hit_rate = (history['Outcome'] == 'Win').mean()
        volatility = history['ProfitLoss'].std()
        returns = history['ProfitLoss'] / history['Stake']
        std_dev_returns = returns.std()

        # Value at Risk (VaR): This is a measure of the risk of potential losses. It's typically calculated as the worst 
        # loss that you could expect to occur with a certain level of confidence (e.g., 95% or 99%) over a certain period. 
        VaR_95 = norm.ppf(1-0.95, np.mean(returns), std_dev_returns)

        running_total = history['ProfitLoss'].cumsum()
        running_max = running_total.cummax()
        drawdown = running_max - running_total
        max_drawdown = drawdown.max()

        return {
            'Total Profit': total_profit,
            'Total Staked': total_staked,
            'Current Bankroll': self.bankroll,
            'Standard Deviation of Returns': std_dev_returns,
            'ROI': ROI,
            'Hit Rate': hit_rate,
            'Volatility': volatility,
            'Max Drawdown': max_drawdown
        }


    @abstractmethod
    def run(self, dataset, model, initial_bankroll: int = 1000):
        pass
