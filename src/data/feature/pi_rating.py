from collections import defaultdict
import pandas as pd
from data.feature.feature_generator import FeatureGenerator


class PiRatingGenerator:
    def __init__(self, alpha: float=0.04, beta: float=0.93):
        self.alpha = alpha
        self.beta = beta

    def generate(self, df: pd.DataFrame):
        df_updated = df.copy()
        pi_ratings = defaultdict(lambda: 1.0)
        home_pi_before = []
        away_pi_before = []
        for i, row in df_updated.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            result = row['FTR']

            home_pi = pi_ratings[home_team]
            away_pi = pi_ratings[away_team]

            home_pi_before.append(home_pi)
            away_pi_before.append(away_pi)

            if result == 'H':
                result_factor = 1
            elif result == 'D':
                result_factor = 0.5
            else:
                result_factor = 0

            delta = self.alpha * (result_factor - home_pi / (home_pi + away_pi**self.beta) if home_pi + away_pi != 0 else 0)
            pi_ratings[home_team] += delta
            pi_ratings[away_team] -= delta

        df_updated['HomePiBefore'] = home_pi_before
        df_updated['AwayPiBefore'] = away_pi_before

        return df_updated
