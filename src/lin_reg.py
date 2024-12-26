import pandas as pd
import numpy as np

class Lin_reg:

    def __init__(self, df: pd.DataFrame, target: str, step: float) -> None:
        self.target = np.array(df[target])
        self.init_coef(df)
        self.step = step
        self.values = np.array(df.drop(columns=[target]))

    def init_coef(self, df: pd.DataFrame) -> None:
        self.coef = np.array([])
        self.coef = np.append(self.coef, np.mean(self.target))
        for i in range (1, df.shape[1]):
            self.coef = np.append(self.coef, 0)
