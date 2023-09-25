from abc import ABC, abstractmethod
from typing import List
from enum import Enum
import pandas as pd
import numpy as np
from math import sqrt
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
    
    def _shin_odds_to_probability(
        odds: List[float],
        max_iterations: int = 1000,
        convergence_threshold: float = 1e-12) -> dict:
        if len(odds) < 2:
            raise ValueError('len(odds) must be >= 2')

        if any(o < 1 for o in odds):
            raise ValueError('All odds must be >= 1')

        z = 0
        n = len(odds)
        inverse_odds = [1.0 / o for o in odds]
        sum_inverse_odds = sum(inverse_odds)
        delta = float('Inf')
        iterations = 0

        if n == 2:
            diff_inverse_odds = inverse_odds[0] - inverse_odds[1]
            z = (
                    ((sum_inverse_odds - 1) * (diff_inverse_odds ** 2 - sum_inverse_odds)) /
                    (sum_inverse_odds * (diff_inverse_odds ** 2 - 1))
            )
            delta = 0
        else:
            while delta > convergence_threshold and iterations < max_iterations:
                z0 = z
                z = (sum(sqrt(z ** 2 + 4 * (1 - z) * io ** 2 / sum_inverse_odds) for io in inverse_odds) - 2) / (n - 2)
                delta = abs(z - z0)
                iterations += 1

        probabilities_adjusted = [(sqrt(z ** 2 + 4 * (1 - z) * io ** 2 / sum_inverse_odds) - z) / (2 * (1 - z)) for io in inverse_odds]
        return probabilities_adjusted
    
    def _expected_value(self, p: float, odds: float) -> float:
        """
        Calculate the expected value of a bet.

        :param p: The estimated probability of winning.
        :param odds: The betting odds.
        :return: The expected value of the bet.
        """
        # potential_profit = odds - 1
        # potential_loss = 1
        # EV = (p * potential_profit) - (p * potential_loss)
        return (p * odds) - 1
    
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
