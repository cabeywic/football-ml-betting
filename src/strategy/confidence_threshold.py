from sklearn.preprocessing import LabelEncoder
from strategy.betting_strategy import BettingStrategy, BetType
import pandas as pd
from typing import List

class ConfidenceThresholdStrategy(BettingStrategy):
    def __init__(self, label_encoder: LabelEncoder, initial_bankroll=1000, bookmakers=['B365', 'IW', 'BW'], confidence_threshold: float=0.6, bet_size: float=0.01):
        super().__init__(initial_bankroll)
        self.confidence_threshold = confidence_threshold
        self.label_encoder = label_encoder
        self.bookmakers = bookmakers
        self.bet_size = bet_size

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

    def _place_bet(self, match: int, match_result: int, bet_type: BetType, prob_pred: float, odds: float):
        # Convert the odds to probabilities
        prob_odds = self._odds_to_probability(odds)

        # Only place a bet if our model's confidence is above the threshold and is greater than the odds-implied probability
        if prob_pred > self.confidence_threshold and prob_pred > prob_odds:
            # Bet a fixed stake
            stake = self.bankroll * self.bet_size
            pred_result = self.label_encoder.transform([bet_type])[0]

            outcome = 'Win' if (match_result == pred_result) else 'Loss'
            profit_loss = stake * (odds - 1) if outcome == 'Win' else -stake
            self.bankroll += profit_loss

            # Update the history
            new_bet = pd.DataFrame([{
                'Match': match, 
                'Bet': bet_type, 
                'Stake': stake,
                'Odds': odds,
                'Predicted Probability': prob_pred,
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
            
            best_prob_pred = max(probabilities_pred)

            # Get the index of the outcome with the highest expected return
            best_prob_pred_idx = probabilities_pred.index(best_prob_pred)

            if best_prob_pred > 0:
                self._place_bet(
                    i, 
                    result.iloc[i], 
                    bet_types[best_prob_pred_idx], 
                    probabilities_pred[best_prob_pred_idx], 
                    odds[best_prob_pred_idx]
                )
