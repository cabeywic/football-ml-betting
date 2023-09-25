from sklearn.preprocessing import LabelEncoder
from strategy.betting_strategy import BettingStrategy, BetType
import pandas as pd
from typing import List


class FixedFractionalStrategy(BettingStrategy):
    """
    A strategy that uses Fixed Fractional Betting, also known as a "fixed wager" strategy

    Fixed Fractional Betting uses a consistent stake size which reduces the risk of huge losses. 
    This strategy is particularly useful when you want to limit the potential losses and maintain a slow and steady bankroll growth.
    """
    def __init__(self, label_encoder: LabelEncoder, initial_bankroll=1000, bookmakers: List[str]=['B365', 'IW', 'BW', 'PS'], fraction: float=0.05):
        super().__init__(initial_bankroll)
        self.fraction = fraction
        self.bookmakers = bookmakers
        self.label_encoder = label_encoder

    def _get_odds(self, match_features: pd.Series, bookmaker: str) -> tuple:
        """
        Retrieve the Home, Draw, and Away odds for a specific bookmaker from a given DataFrame match_features.

        :param match_features: The match_features in the DataFrame representing a match.
        :param bookmaker: The bookmaker's code (e.g., 'B365').
        :return: A tuple containing the odds for Home, Draw, and Away, in that order.
        """
        home_odds = match_features[f'{bookmaker}H']
        draw_odds = match_features[f'{bookmaker}D']
        away_odds = match_features[f'{bookmaker}A']
        
        return home_odds, draw_odds, away_odds

    def _get_best_odds(self, match_features: pd.Series) -> tuple:
        """
        Get the best odds for a match across multiple bookmakers.

        :param match_features: The features of the match.
        :return: The best odds for home win, draw, and away win.
        """
        bookmakers = self.bookmakers
        best_odds_home = max(match_features[[bookmaker + 'H' for bookmaker in bookmakers]])
        best_odds_draw = max(match_features[[bookmaker + 'D' for bookmaker in bookmakers]])
        best_odds_away = max(match_features[[bookmaker + 'A' for bookmaker in bookmakers]])
        return best_odds_home, best_odds_draw, best_odds_away

    def _place_bet(self, match: int, match_result: int, bet_type: BetType, odds: float):
        stake = self.fraction * self.bankroll
        pred_result = self.label_encoder.transform([bet_type])[0]

        outcome = 'Win' if (match_result == pred_result) else 'Loss'
        profit_loss = stake * (odds - 1) if outcome == 'Win' else -stake
        self.bankroll += profit_loss

        new_bet = pd.DataFrame([{
            'Match': match, 
            'Bet': bet_type, 
            'Stake': stake, 
            'Odds': odds, 
            'Outcome': outcome, 
            'ProfitLoss': profit_loss, 
            'Bankroll': self.bankroll
        }])
            
        self.history = pd.concat([self.history, new_bet], ignore_index=True)

    def run(self, features: pd.DataFrame, result: pd.Series, model):
        bet_types = [BetType.HOME.value, BetType.DRAW.value, BetType.AWAY.value]
        
        for i in range(len(features)):
            odds_home, odds_draw, odds_away = self._get_best_odds(features.iloc[i])
            odds = [odds_home, odds_draw, odds_away]
            
            probabilities_pred = [model.predict_proba(features.iloc[i:i+1])[0][bet_type] for bet_type in self.label_encoder.transform(bet_types)]
            expected_returns = [self._expected_value(prob_pred, odd) for prob_pred, odd in zip(probabilities_pred, odds)]
            
            best_expected_return = max(expected_returns)

            # Get the index of the outcome with the highest expected return
            best_outcome_index = expected_returns.index(best_expected_return)

            if best_expected_return > 0:
                # Place a bet on this outcome
                # print("ODDS: ", odds)
                # print("Pred Prob: ", probabilities_pred)
                # print("Expected Return: ", expected_returns)
                # print("Best Outcome: ", bet_types[best_outcome_index], result.iloc[i])
                # print("-"*50)
                self._place_bet(
                    i, 
                    result.iloc[i], 
                    bet_types[best_outcome_index], 
                    odds[best_outcome_index]
                )
