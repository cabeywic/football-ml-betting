from sklearn.preprocessing import LabelEncoder
from strategy.betting_strategy import BettingStrategy, BetType
import pandas as pd
from typing import List


class KellyCriterionStrategy(BettingStrategy):
    """
    A strategy that uses the Kelly Criterion to determine the bet size. 

    The Kelly Criterion determines the bet size as a fraction of the available bankroll, based on the estimated 
    probability of winning and the odds offered by the bookmaker. 

    The strategy places a bet if the estimated probability of winning is greater than the odds-implied probability. 
    """
    def __init__(self, label_encoder: LabelEncoder, initial_bankroll: int=1000, bookmakers: List[str]=['B365', 'IW', 'BW'], bet_on_all: bool=False):
        """
        Initialize the KellyCriterionStrategy.

        :param label_encoder: A LabelEncoder to transform the match results into numerical format.
        :param initial_bankroll: The initial bankroll available for betting.
        :param bookmakers: A list of bookmakers to consider when placing bets.
        """
        super().__init__(initial_bankroll)
        self.label_encoder = label_encoder
        self.bookmakers = bookmakers
        self.bet_on_all = bet_on_all
    
    def _place_bet(self, match: int, match_result: int, bet_type: BetType, prob_pred: float, odds: float):
        """
        Place a bet based on the Kelly Criterion.

        :param match: The match index.
        :param match_result: The result of the match.
        :param bet_type: The type of bet to place (home win, draw, away win).
        :param prob_pred: The estimated probability of winning the bet.
        :param odds: The betting odds.
        """
        # Convert the odds to probabilities
        prob_odds = self._odds_to_probability(odds)

        # If the estimated probability is greater than the odds-implied probability, place a bet
        if prob_pred > prob_odds:
            stake = self._kelly_criterion(prob_pred, odds) * self.bankroll
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
                'Actual Probability': prob_odds,
                'Outcome': outcome, 
                'ProfitLoss': profit_loss, 
                'Bankroll': self.bankroll
            }])
            
            self.history = pd.concat([self.history, new_bet], ignore_index=True)

    def _kelly_criterion(self, p: float, r: float) -> float:
        """
        Calculate the bet size using the Kelly Criterion.

        :param p: The estimated probability of winning.
        :param r: The betting odds.
        :return: The fraction of the bankroll to bet.
        """
        return max((r * p - 1) / (r - 1), 0)
    
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
    
    def run(self, features: pd.DataFrame, result: pd.Series, model):
        """
        Run the Kelly Criterion Strategy on a dataset.

        :param features: The features of the matches.
        :param result: The results of the matches.
        :param model: The predictive model to use.
        """
        # For each match in the test set
        for i in range(len(features)):
            # Get the odds
            odds_home, odds_draw, odds_away = self._get_best_odds(features.iloc[i])
            odds = [odds_home, odds_draw, odds_away]
            bet_types = [BetType.HOME.value, BetType.DRAW.value, BetType.AWAY.value]

            probabilities_pred = [model.predict_proba(features.iloc[i:i+1])[0][bet_type] for bet_type in self.label_encoder.transform(bet_types)]
            expected_returns = [self._expected_value(prob_pred, odd) for prob_pred, odd in zip(probabilities_pred, odds)]

            if self.bet_on_all:
                for j in range(len(expected_returns)):
                    self._place_bet(i, result.iloc[i], bet_types[j], probabilities_pred[j], odds[j])
            else:
                # Get the index of the outcome with the highest expected return
                best_outcome_index = expected_returns.index(max(expected_returns))

                # Place a bet on this outcome
                self._place_bet(
                    i, 
                    result.iloc[i], 
                    bet_types[best_outcome_index], 
                    probabilities_pred[best_outcome_index], 
                    odds[best_outcome_index]
                )


class FractionalKellyCriterionStrategy(KellyCriterionStrategy):
    """
    A betting strategy that uses a fractional version of the Kelly Criterion to determine the bet size.
    
    The Kelly Criterion calculates the optimal bet size as a fraction of the bankroll, based on the estimated 
    probability of winning and the odds offered by the bookmaker. The traditional Kelly Criterion tends to recommend 
    large bets, which can lead to substantial losses in case of a losing streak, potentially exhausting the bankroll.
    
    This fractional version of the Kelly Criterion aims to mitigate this risk. It does so by reducing the bet size 
    as a fixed fraction of the size recommended by the Kelly Criterion. The fraction is a parameter that can be set 
    when initializing the strategy. A smaller fraction will lead to smaller bets and a more cautious strategy, while 
    a larger fraction will lead to larger bets and a more aggressive strategy.
    
    By reducing the bet size, this strategy aims to reduce the risk of large losses and make the bankroll last longer.
    It trades off some potential return in order to decrease the risk of ruin, providing a more conservative approach 
    to betting.
    """

    def __init__(self, label_encoder: LabelEncoder, initial_bankroll=1000, bookmakers=['B365', 'IW', 'BW'], fraction: float=0.05):
        super().__init__(label_encoder, initial_bankroll, bookmakers)
        self.fraction = fraction

    def _kelly_criterion(self, p: float, r: float) -> float:
        """
        Calculate the bet size using the Kelly Criterion, and multiply by a fraction to reduce risk.

        :param p: The estimated probability of winning.
        :param r: The betting odds.
        :return: The fraction of the bankroll to bet.
        """
        kelly_fraction = super()._kelly_criterion(p, r)
        return self.fraction * kelly_fraction


class AdaptiveFractionalKellyCriterionStrategy(FractionalKellyCriterionStrategy):
    def __init__(self, label_encoder: LabelEncoder, initial_bankroll=1000, bookmakers=['B365', 'IW', 'BW'], base_fraction: float=0.05, window_size=5):
        super().__init__(label_encoder, initial_bankroll, bookmakers, base_fraction)
        self.base_fraction = base_fraction
        self.window_size = window_size  # the number of recent games to consider for adjusting the fraction

    def _calculate_fraction(self):
        """
        Adjust the fraction based on recent performance.

        Increase the fraction if the model has been performing well recently, decrease it if the model has been performing poorly.
        """
        if len(self.history) > 0:
            recent_bets = self.history.tail(min(self.window_size, len(self.history)))
            win_rate = recent_bets[recent_bets['Outcome'] == 'Win'].shape[0] / len(recent_bets)
            return self.base_fraction * (1 + win_rate)  # increase the fraction if win_rate > 0, decrease it if win_rate < 0
        else:
            return self.base_fraction


    def _kelly_criterion(self, p: float, r: float) -> float:
        """
        Calculate the bet size using the Kelly Criterion, and multiply by a dynamically calculated fraction to reduce risk.

        :param p: The estimated probability of winning.
        :param r: The betting odds.
        :return: The fraction of the bankroll to bet.
        """
        self.fraction = self._calculate_fraction()
        return super()._kelly_criterion(p, r)


class DynamicFractionalKellyCriterionStrategy(KellyCriterionStrategy):
    """
    A betting strategy that uses a dynamic, fractional version of the Kelly Criterion to determine the bet size.
    
    The traditional Kelly Criterion calculates the optimal bet size as a fraction of the bankroll, based on the 
    estimated probability of winning and the odds offered by the bookmaker. It tends to recommend large bets, 
    which can lead to substantial losses in case of a losing streak, potentially exhausting the bankroll.
    
    This dynamic, fractional version of the Kelly Criterion aims to mitigate this risk. It does so by adjusting the 
    fraction of the bankroll to bet based on the size of the bankroll and the perceived risk of the bet.
    
    The strategy sets a lower and an upper threshold for the bankroll. When the bankroll falls below the lower 
    threshold, the strategy becomes more cautious and reduces the bet size. Conversely, when the bankroll rises 
    above the upper threshold, the strategy becomes more aggressive and increases the bet size.
    
    The perceived risk of a bet is calculated as the absolute difference between the predicted probability and the 
    odds-implied probability. The greater this difference, the greater the perceived risk. The strategy adjusts the 
    bet size inversely to the perceived risk, betting less on riskier bets and more on less risky bets.
    
    This approach provides a balance between risk and return, aiming to maximize long-term growth of the bankroll 
    while reducing the risk of ruin.
    """
    def __init__(self, label_encoder: LabelEncoder, initial_bankroll=1000, bookmakers=['B365', 'IW', 'BW'], low_threshold: float=200, high_threshold: float=2000, min_fraction: float=0.05, max_fraction: float=0.25):
        super().__init__(label_encoder, initial_bankroll, bookmakers)
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction

    def _get_fraction(self, prob_pred: float, prob_odds: float) -> float:
        """
        Calculate the fraction to use based on the current bankroll and the perceived risk of the bet.

        :param prob_pred: The predicted probability of winning.
        :param prob_odds: The odds-implied probability.
        :return: The fraction to use.
        """
        # Determine the base fraction based on the current bankroll
        if self.bankroll > self.high_threshold:
            base_fraction = self.max_fraction
        elif self.bankroll < self.low_threshold:
            base_fraction = self.min_fraction
        else:
            # Calculate the base fraction for betting, which scales linearly from min_fraction to max_fraction as our 
            # bankroll increases from low_threshold to high_threshold. This allows us to bet more when our bankroll is 
            # larger and less when our bankroll is smaller.
            base_fraction = ((self.bankroll - self.low_threshold) / (self.high_threshold - self.low_threshold)) * (self.max_fraction - self.min_fraction) + self.min_fraction

        # Adjust the fraction based on the perceived risk of the bet
        risk_adjustment = abs(prob_pred - prob_odds)
        fraction = base_fraction * risk_adjustment

        return fraction

    def _kelly_criterion(self, p: float, r: float) -> float:
        """
        Calculate the bet size using the Kelly Criterion, and multiply by a dynamic fraction to reduce risk.

        :param p: The estimated probability of winning.
        :param r: The betting odds.
        :return: The fraction of the bankroll to bet.
        """
        kelly_fraction = super()._kelly_criterion(p, r)
        dynamic_fraction = self._get_fraction(p, 1/r)
        return dynamic_fraction * kelly_fraction
