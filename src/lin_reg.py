import pandas as pd
import numpy as np

class Lin_reg:

    def __init__(self, df: pd.DataFrame, target: str, step: float) -> None:
        self.target = np.array(df[target])
        self.init_coef(df)
        self.step = step
        self.values = np.array(df.drop(columns=[target]))
        self.values = np.c_[np.ones(self.values.shape[0]), self.values]
        self.n = df.shape[0]

    def init_coef(self, df: pd.DataFrame) -> None:
        self.coef = np.array([])
        self.coef = np.append(self.coef, np.mean(self.target))
        for i in range (1, df.shape[1]):
            self.coef = np.append(self.coef, 0)

    def predict(self) -> float:
        return np.dot(self.values, self.coef)

    def MSE(self) -> float:
        return 1/self.n * float(np.sum(self.target - self.predict()))**2
