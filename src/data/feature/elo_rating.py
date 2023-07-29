from collections import defaultdict
import pandas as pd
from data.feature.feature_generator import FeatureGenerator


class EloRatingGenerator(FeatureGenerator):
    def __init__(self, K: int = 20) -> None:
        self.elo_ratings = defaultdict(lambda: 1500)
        self.K = K

    def _expected_score(self, R1, R2):
        return 1 / (1 + 10 ** ((R2 - R1) / 400))
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df_updated = df.copy()
        
        K = self.K
        home_elo_ratings = []
        away_elo_ratings = []
        
        for i, row in df_updated.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            outcome = row['FTR']

            # Get the ELO ratings of the teams before the match
            home_elo_before = self.elo_ratings[home_team]
            away_elo_before = self.elo_ratings[away_team]

            # Calculate the expected outcomes
            expected_home = self._expected_score(home_elo_before, away_elo_before)
            expected_away = 1 - expected_home

            # Update the ELO ratings based on the actual outcome
            if outcome == 'H':  # Home win
                self.elo_ratings[home_team] = home_elo_before + K * (1 - expected_home)
                self.elo_ratings[away_team] = away_elo_before + K * (0 - expected_away)
            elif outcome == 'D':  # Draw
                self.elo_ratings[home_team] = home_elo_before + K * (0.5 - expected_home)
                self.elo_ratings[away_team] = away_elo_before + K * (0.5 - expected_away)
            elif outcome == 'A':  # Away win
                self.elo_ratings[home_team] = home_elo_before + K * (0 - expected_home)
                self.elo_ratings[away_team] = away_elo_before + K * (1 - expected_away)

            # Add the ELO ratings before the match to the lists
            home_elo_ratings.append(home_elo_before)
            away_elo_ratings.append(away_elo_before)

        # Add the ELO ratings as features to the dataframe
        df_updated['HomeEloBefore'] = home_elo_ratings
        df_updated['AwayEloBefore'] = away_elo_ratings
        
        return df_updated