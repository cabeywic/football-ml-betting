import os
import logging
import pandas as pd
from typing import List, Optional, Dict
from dataclasses import dataclass
from utils import parse_date
from data.feature.feature_generator import FeatureGenerator


@dataclass
class Division:
    name: str
    dataframe: pd.DataFrame

    def compute_features(self, feature_generators: List[FeatureGenerator]):
        for generator in feature_generators:
            self.dataframe = generator.generate(self.dataframe)

@dataclass
class FootballData:
    columns: List[str]
    divisions: Dict[str, Division]

    def _logger(self):
        return logging.getLogger(__name__)

    def compute_features(self, feature_generators: List[FeatureGenerator]):
        self._logger().info(f'Computing features for {len(self.divisions)} divisions...')
        for division in self.divisions.values():
            division.compute_features(feature_generators)
        self._logger().info('Successfully computed features')
    
    @staticmethod
    def load(path: str, columns: List[str], years: Optional[List[str]] = None, division_names: Optional[List[str]] = None) -> 'FootballData':
        if years is None:
            years = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
        
        years.sort()
        # print(f'Loading data for years: {years}\n\n')

        divisions = {}
        for year in years:
            year_path = os.path.join(path, year)
            if division_names is None:
                division_files = [f for f in os.listdir(year_path) if f.endswith('.csv')]
                division_names = [os.path.splitext(f)[0] for f in division_files]
                
            # print(f'Loading data [{year}] for divisions: {division_names}')
            for division_name in division_names:
                file_path = os.path.join(year_path, f'{division_name}.csv')
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, usecols=columns, encoding='ISO-8859-1')
                    # df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
                    df['Date'] = df['Date'].apply(parse_date)
                    df.dropna(how='all', inplace=True)
                    df['Season'] = year
                    if division_name in divisions.keys():
                        divisions[division_name].dataframe = pd.concat([divisions[division_name].dataframe, df], ignore_index=True)
                    else:
                        division = Division(name=division_name, dataframe=df)
                        divisions[division_name] = division
        return FootballData(columns=columns, divisions=divisions)
    