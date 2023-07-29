from collections import defaultdict
import pandas as pd
from data.feature.feature_generator import FeatureGenerator


class GapRatingGenerator():
    def __init__(self, lambda_: float=0.3, phi1: float=0.4, phi2: float=0.6, input_features = [("FTHG", "FTAG")]) -> None:
        self.lambda_ = lambda_
        self.phi1 = phi1
        self.phi2 = phi2
        self.input_features = input_features

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df_updated = df.copy()
        lambda_ = self.lambda_
        phi1 = self.phi1
        phi2 = self.phi2

        for home_feature, away_feature in self.input_features:
            df_updated[f"{home_feature}_norm"] = df_updated[home_feature] / df_updated[home_feature].max()
            df_updated[f"{away_feature}_norm"] = df_updated[away_feature] / df_updated[away_feature].max()

        df_updated['HomeAttack'] = 0
        df_updated['AwayAttack'] = 0

        for home_feature, away_feature in self.input_features:
            df_updated['HomeAttack'] += df_updated[f"{home_feature}_norm"]
            df_updated['AwayAttack'] += df_updated[f"{away_feature}_norm"]

        # Initialize dictionaries to hold the GAP ratings for each team
        gap_ratings_home_attack = defaultdict(lambda: df_updated['HomeAttack'].mean())
        gap_ratings_home_defend = defaultdict(lambda: df_updated['AwayAttack'].mean())
        gap_ratings_away_attack = defaultdict(lambda: df_updated['AwayAttack'].mean())
        gap_ratings_away_defend = defaultdict(lambda: df_updated['HomeAttack'].mean())

        # Initialize lists to hold the GAP ratings for each match
        home_gap_attack_before = []
        home_gap_defend_before = []
        away_gap_attack_before = []
        away_gap_defend_before = []

        # Parameters for the GAP rating calculation
        

        # Iterate over the matches in the dataframe
        for i, row in df_updated.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            goals_home = row['FTHG']
            goals_away = row['FTAG']

            # Get the GAP ratings of the teams before the match
            home_gap_attack = gap_ratings_home_attack[home_team]
            home_gap_defend = gap_ratings_home_defend[home_team]
            away_gap_attack = gap_ratings_away_attack[away_team]
            away_gap_defend = gap_ratings_away_defend[away_team]

            # Update the GAP ratings
            gap_ratings_home_attack[home_team] = max(home_gap_attack + lambda_ * phi1 * (goals_home - (home_gap_attack + away_gap_defend) / 2), 0)
            gap_ratings_home_defend[home_team] = max(home_gap_defend + lambda_ * phi1 * (goals_away - (away_gap_attack + home_gap_defend) / 2), 0)
            gap_ratings_away_attack[away_team] = max(away_gap_attack + lambda_ * phi2 * (goals_away - (away_gap_attack + home_gap_defend) / 2), 0)
            gap_ratings_away_defend[away_team] = max(away_gap_defend + lambda_ * phi2 * (goals_home - (home_gap_attack + away_gap_defend) / 2), 0)

            # Also update the away ratings of the home team and the home ratings of the away team
            gap_ratings_away_attack[home_team] = max(gap_ratings_away_attack[home_team] + lambda_ * (1 - phi1) * (goals_home - (home_gap_attack + away_gap_defend) / 2), 0)
            gap_ratings_away_defend[home_team] = max(gap_ratings_away_defend[home_team] + lambda_ * (1 - phi1) * (goals_away - (away_gap_attack + home_gap_defend) / 2), 0)
            gap_ratings_home_attack[away_team] = max(gap_ratings_home_attack[away_team] + lambda_ * (1 - phi2) * (goals_away - (away_gap_attack + home_gap_defend) / 2), 0)
            gap_ratings_home_defend[away_team] = max(gap_ratings_home_defend[away_team] + lambda_ * (1 - phi2) * (goals_home - (home_gap_attack + away_gap_defend) / 2), 0)

            # Add the GAP ratings before the match to the lists
            home_gap_attack_before.append(home_gap_attack)
            home_gap_defend_before.append(home_gap_defend)
            away_gap_attack_before.append(away_gap_attack)
            away_gap_defend_before.append(away_gap_defend)

        # Add the GAP ratings as features to the dataframe
        df_updated['HomeGapAttackBefore'] = home_gap_attack_before
        df_updated['HomeGapDefendBefore'] = home_gap_defend_before
        df_updated['AwayGapAttackBefore'] = away_gap_attack_before
        df_updated['AwayGapDefendBefore'] = away_gap_defend_before

        return df_updated
