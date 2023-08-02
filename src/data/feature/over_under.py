import pandas as pd
from typing import List
from data.feature.feature_generator import FeatureGenerator


class OverUnderGenerator(FeatureGenerator):
    def __init__(self, over_under_values: List[float] = [0.5, 1.5, 2.5]):
        super().__init__()
        self.over_under_values = over_under_values

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df_updated = df.copy()
        
        # Calculate full time goals
        df_updated['FTG'] = df_updated['FTHG'] + df_updated['FTAG']
        
        # Generate over/under columns
        for value in self.over_under_values:
            df_updated[f'FTG>{value}'] = df_updated['FTG'].apply(lambda x: 1 if x > value else 0)

        return df_updated
