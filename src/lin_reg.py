import pandas as pd
import numpy as np

class Lin_reg:

    def __init__(self, df: pd.DataFrame, target: str, step: float):
        self.coef = np.array([])
        self.coef = np.append(self.coef, np.mean(df[target]))
        for i in range (1, df.shape[1]):
            self.coef = np.append(self.coef, 0)
        self.step = step
