from collections import defaultdict
import pandas as pd
from data.feature.feature_generator import FeatureGenerator


class WinStreakGenerator(FeatureGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.win_streaks = defaultdict(int)

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df_updated = df.copy()
        home_win_streaks = []
        away_win_streaks = []
        
        for i, row in df_updated.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            outcome = row['FTR']

            home_streak_before = self.win_streaks[home_team]
            away_streak_before = self.win_streaks[away_team]

            # Update the win streaks based on the actual outcome
            if outcome == 'H':  # Home win
                self.win_streaks[home_team] = home_streak_before + 1
                self.win_streaks[away_team] = 0
            elif outcome == 'D':  # Draw
                self.win_streaks[home_team] = 0
                self.win_streaks[away_team] = 0
            elif outcome == 'A':  # Away win
                self.win_streaks[home_team] = 0
                self.win_streaks[away_team] = away_streak_before + 1

            # Add the win streaks before the match to the lists
            home_win_streaks.append(home_streak_before)
            away_win_streaks.append(away_streak_before)

        df_updated['HomeStreakBefore'] = home_win_streaks
        df_updated['AwayStreakBefore'] = away_win_streaks

        return df_updated