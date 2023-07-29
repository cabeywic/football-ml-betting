from abc import ABC, abstractmethod
import pandas as pd


class FeatureGenerator(ABC):
    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
